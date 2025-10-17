import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
from collections import deque
import math






#  Epsilon controller (panic bump). basically the modifitcaiton from the baseline, so that my strategy works
class EpsilonController:
    def __init__(self, base_schedule="exp",
                 eps_start=0.5, eps_min=0.02, max_cap=0.5,
                 decay_steps=60_000, tau=30_000,
                 bad_bar=120, bump=0.08,
                 decay_back=0.97, cooldown_eps=5):
        self.base = base_schedule
        self.start, self.min, self.max_cap = eps_start, eps_min, max_cap
        self.decay_steps, self.tau = decay_steps, tau
        self.bad_bar, self.bump = bad_bar, bump
        self.decay_back, self.cooldown_eps = decay_back, cooldown_eps
        self.step = 0
        self.eps = self.start
        self._cooldown = 0

    def _base_eps(self):
        if self.base == "exp":
            return self.min + (self.start - self.min) * math.exp(-self.step / self.tau)
        frac = min(1.0, self.step / self.decay_steps)
        return max(self.min, self.start - (self.start - self.min) * frac)

    def on_step(self):
        self.step += 1
        self.eps = max(self.min, min(self.max_cap, self._base_eps()))
        return self.eps

    def on_episode_end(self, total_reward):
        self.eps = max(self.min, min(self.max_cap, self.eps * self.decay_back))
        if self._cooldown > 0:
            self._cooldown -= 1
            return self.eps
        if total_reward < self.bad_bar:
            self.eps = min(self.max_cap, self.eps + self.bump)
            self._cooldown = self.cooldown_eps
        return self.eps

#  Networks & Replay 
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class BaseMemoryBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        import random
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return (
            np.float32(state), np.int64(action), np.float32(reward),
            np.float32(next_state), np.float32(done)
        )

# DQN Wrapper 
class BaseDeepQModel(nn.Module):
    def __init__(self, state_dim, action_dim,
                 memory_buffer_size=10_000,
                 learning_rate=1e-4,
                 discount_factor=0.8):
        super().__init__()
        self.memory_buffer = BaseMemoryBuffer(memory_buffer_size)
        self.QNetwork = QNetwork(state_dim, action_dim)
        self.TNetwork = QNetwork(state_dim, action_dim)
        self.TNetwork.load_state_dict(self.QNetwork.state_dict())
        self.optimizer = optim.Adam(self.QNetwork.parameters(), lr=learning_rate)
        self.gamma = discount_factor
        self.env = gym.make("CartPole-v1")

    def update_pole_length(self, value):
        self.env.unwrapped.length = float(value)

    def save_model(self, model_name):
        torch.save(self.QNetwork.state_dict(), f"./models/{model_name}.pth")

    def update_target_network(self):
        self.TNetwork.load_state_dict(self.QNetwork.state_dict())

    def calculate_loss(self, batch_size=256):
        if len(self.memory_buffer.buffer) < batch_size:
            return None
        states, actions, rewards, next_states, dones = self.memory_buffer.sample(batch_size)
        states      = torch.as_tensor(states, dtype=torch.float32)
        actions     = torch.as_tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards     = torch.as_tensor(rewards, dtype=torch.float32)
        next_states = torch.as_tensor(next_states, dtype=torch.float32)
        dones       = torch.as_tensor(dones, dtype=torch.float32)

        # Q(s,a)
        q_values = self.QNetwork(states)
        q_sa = q_values.gather(1, actions).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            next_q_online = self.QNetwork(next_states)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.TNetwork(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + self.gamma * next_q_target * (1.0 - dones)

        return nn.MSELoss()(q_sa, targets)

    def choose_action(self, observation, epsilon, warmup=False):
        if warmup or (epsilon is not None and random.random() < epsilon):
            return self.env.action_space.sample()
        with torch.no_grad():
            q = self.QNetwork(torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0))
            return int(q.argmax(dim=1).item())

    def train_loop(self, model_name,
                   episodes_per_length=500,
                   t_net_update_freq=50,
                   sample_batch=256,
                   warmup_steps=1000,
                   pole_lengths=None):

        eps_ctl = EpsilonController(
            base_schedule="exp",
            eps_start=0.5, eps_min=0.02, max_cap=0.5,
            tau=30_000, bad_bar=120, bump=0.08,
            decay_back=0.97, cooldown_eps=5
        )

        if pole_lengths is None:
            pole_lengths = np.linspace(0.4, 1.8, 30)  # assignment grid

        episode_rewards = []
        total_steps = 0
        warmup = True

        for L in pole_lengths:
            self.update_pole_length(L)
            print(f"\n=== Training on pole length {L:.2f} ===")

            for episode in range(episodes_per_length):
                if warmup and total_steps > warmup_steps:
                    warmup = False

                state, _ = self.env.reset()
                done = False
                ep_reward = 0

                while not done:
                    eps = 0.5 if warmup else eps_ctl.on_step()
                    action = self.choose_action(state, epsilon=eps, warmup=warmup)
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated

                    self.memory_buffer.push((state, action, reward, next_state, float(done)))
                    state = next_state
                    ep_reward += reward
                    total_steps += 1

                    if warmup:
                        continue

                    loss = self.calculate_loss(sample_batch)
                    if loss is not None:
                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.QNetwork.parameters(), 10.0)
                        self.optimizer.step()

                if not warmup:
                    eps_ctl.on_episode_end(ep_reward)

                episode_rewards.append(ep_reward)
                avg50 = sum(episode_rewards[-50:]) / min(50, len(episode_rewards))
                print(f"\rLen {L:.2f} | Ep {episode+1:4d}/{episodes_per_length} | "
                      f"R={ep_reward:4.0f} | eps={eps_ctl.eps: .3f} | avg50={avg50:6.1f}",
                      end="", flush=True)

                if (episode + 1) % t_net_update_freq == 0:
                    self.update_target_network()

        self.save_model(model_name)

if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    model = BaseDeepQModel(
        state_dim=4, action_dim=2,
        memory_buffer_size=10_000,
        learning_rate=1e-4,
        discount_factor=0.8
    )

    #  testing the entire scenario in here
    model.train_loop(
        model_name="dqn_eps_panicbump_smoke",
        episodes_per_length=200,
        t_net_update_freq=50,
        sample_batch=256,
        warmup_steps=500,
        pole_lengths=[0.4, 1.1, 1.6]  #  test (3 lengths) — comment this line and use the full grid when you’re ready
    )
