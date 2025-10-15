import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
from collections import deque
import math





class EpsilonController:
    def __init__(self,
                 base_schedule="linear",  # "linear" or "exp"
                 eps_min=0.05, eps_max=1.0,
                 decay_steps=200_000,     # for linear
                 tau=100_000,             # for exp
                 bad_bar=200,             # episode reward threshold
                 bump=0.20,               # epsilon increase after bad ep
                 decay_back=0.98,         # per-episode shrink toward base
                 cooldown_eps=3):         # avoid yo-yo zzz
        self.base = base_schedule
        self.min, self.max = eps_min, eps_max
        self.decay_steps, self.tau = decay_steps, tau
        self.bad_bar, self.bump = bad_bar, bump
        self.decay_back, self.cooldown_eps = decay_back, cooldown_eps
        self.step = 0
        self.eps = self.max
        self._cooldown = 0

    def _base_eps(self):
        if self.base == "exp":
            return self.min + (self.max - self.min) * math.exp(-self.step / self.tau)
        frac = min(1.0, self.step / self.decay_steps)
        return max(self.min, self.max - (self.max - self.min) * frac)

    def on_step(self):
        self.step += 1
        base = self._base_eps()
        self.eps = max(self.min, min(self.max, 0.9 * self.eps + 0.1 * base))
        return self.eps

    def on_episode_end(self, total_reward):
        self.eps = max(self.min, self.eps * self.decay_back)
        if self._cooldown > 0:
            self._cooldown -= 1
            return self.eps
        if total_reward < self.bad_bar:
            self.eps = min(self.max, self.eps + self.bump)
            self._cooldown = self.cooldown_eps
        return self.eps

# This is the 'standard' neural network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class BaseMemoryBuffer:
    def __init__(self, buffer_size):
        super(BaseMemoryBuffer, self).__init__()
        # Memory buffer for storing experiences
        self.buffer = deque(maxlen=buffer_size)

    def push(self, transition):
        # add new state transition to the buffer, tuple (state, action, reward, state+1, done)
        self.buffer.append(transition)

    def sample(self, batch_size):
        # Sample randomly from the buffer, this can be overwritten to adjust the buffer behaviour
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))  # Unzip to arrays
        return (
            np.float32(state), np.int64(action), np.float32(reward),
            np.float32(next_state), np.float32(done)
        )


class BaseDeepQModel(nn.Module):
    def __init__(self, state_dim, action_dim, memory_buffer_size, learning_rate, discount_factor):
        super(BaseDeepQModel, self).__init__()
        self.memory_buffer = BaseMemoryBuffer(memory_buffer_size)
        self.QNetwork = QNetwork(state_dim, action_dim)
        self.TNetwork = QNetwork(state_dim, action_dim)
        self.TNetwork.load_state_dict(self.QNetwork.state_dict())
        self.optimizer = optim.Adam(self.QNetwork.parameters(), lr=learning_rate)
        self.gamma = discount_factor
        self.env = gym.make("CartPole-v1")  # no render during training


    def update_pole_length(self, value):
        # Change the pole length in the cartpole env
        self.env.unwrapped.length = value

    def save_model(self, model_name):
        # Save your model after training is complete to the models folder
        torch.save(self.QNetwork.state_dict(), f"./models/{model_name}.pth")

    def update_target_network(self):
        self.TNetwork.load_state_dict(self.QNetwork.state_dict())


    def calculate_loss(self, batch_size=64):
        # Calculate MSE loss
        if len(self.memory_buffer.buffer) < batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.memory_buffer.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Current Q(s,a)
        current_q_values = self.QNetwork(states)
        current_q = current_q_values.gather(1, actions).squeeze(1)

        # Target: Double DQN
        with torch.no_grad():
            next_q_online = self.QNetwork(next_states)
            next_actions = next_q_online.max(1)[1].unsqueeze(1)  # Argmax from online
            next_q_target = self.TNetwork(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + self.gamma * next_q_target * (1 - dones)

        # MSE Loss
        loss = nn.MSELoss()(current_q, targets)
        return loss


    def choose_action(self, observation, epsilon, warmup=False, **kwargs):
        if warmup or (epsilon is not None and random.random() < epsilon):
            return self.env.action_space.sample()
        with torch.no_grad():
            q_values = self.QNetwork(torch.tensor(observation, dtype=torch.float32).unsqueeze(0))
            return int(q_values.argmax(dim=1).item())


    def train_loop(self, model_name,
                episodes_per_length=300,
                t_net_update_freq=50, sample_batch=64,
                warmup_steps=1000,
                pole_lengths=None,     # <--- NEW
                **kwargs):

        eps_ctl = EpsilonController(
            base_schedule="linear",
            eps_min=0.05, eps_max=1.0,
            decay_steps=200_000, tau=100_000,
            bad_bar=200, bump=0.20, decay_back=0.98, cooldown_eps=3
        )

        # default to the full 30 if not provided
        if pole_lengths is None:
            pole_lengths = np.linspace(0.4, 1.8, 30)
        else:
            pole_lengths = np.array(pole_lengths, dtype=float)

        episode_rewards = []
        total_steps = 0
        warmup = True

        for L in pole_lengths:
            self.update_pole_length(float(L))
            print(f"\n=== Training on pole length {L:.2f} ===")

            for episode in range(episodes_per_length):
                if warmup and total_steps > warmup_steps:
                    warmup = False

                state, _ = self.env.reset()
                done = False
                ep_reward = 0

                while not done:
                    eps = 1.0 if warmup else eps_ctl.on_step()
                    action = self.choose_action(state, epsilon=eps, warmup=warmup)
                    next_state, reward, done, _, _ = self.env.step(action)

                    self.memory_buffer.push((state, action, reward, next_state, done))
                    state = next_state
                    ep_reward += reward
                    total_steps += 1

                    if warmup:
                        continue

                    loss = self.calculate_loss(sample_batch)
                    if loss is not None:
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

            if not warmup:
                eps_ctl.on_episode_end(ep_reward)

            episode_rewards.append(ep_reward)
            avg50 = sum(episode_rewards[-50:]) / min(50, len(episode_rewards))
            print(f"\rLength {L:.2f} | Ep {episode+1:4d}/{episodes_per_length} | "
                f"R={ep_reward:4.0f} | eps={eps_ctl.eps: .3f} | avg50={avg50:6.1f}",
                end="", flush=True)

            if (episode + 1) % t_net_update_freq == 0:
                self.update_target_network()

        self.save_model(model_name)




if __name__ == "__main__":
    import os, random
    os.makedirs("models", exist_ok=True)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    model = BaseDeepQModel(state_dim=4, action_dim=2,
                           memory_buffer_size=10_000,
                           learning_rate=1e-3,
                           discount_factor=0.99)

    model.train_loop(
        model_name="dqn_eps_panicbump_3lens",
        episodes_per_length=100,         # quick check
        t_net_update_freq=50,
        sample_batch=64,
        warmup_steps=500,
        pole_lengths=[0.6, 1.1, 1.6]     # <- only on 3 lengths
    )

