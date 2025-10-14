import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
from collections import deque


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
    def __init__(self, state_dim, action_dim, memory_buffer_size, learning_rate):
        super(BaseDeepQModel, self).__init__()
        self.memory_buffer = BaseMemoryBuffer(memory_buffer_size)
        self.QNetwork = QNetwork(state_dim, action_dim)
        self.TNetwork = QNetwork(state_dim, action_dim)
        self.TNetwork.load_state_dict(self.QNetwork.state_dict())
        self.optimizer = optim.Adam(self.QNetwork.parameters(), lr=learning_rate)
        self.env = gym.make("CartPole-v1", render_mode="human")

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
        if warmup:
            return random.choice([0, 1])

        if epsilon:
            if random.random() < epsilon:
                return random.choice([0,1])

        probs = self.QNetwork.forward(torch.tensor(observation, dtype=torch.float32))
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return action


    def train_loop(self, model_name, episodes, epsilon=None,
                   t_net_update_freq=50, sample_batch=64,
                   warmup_steps=1000, epsilon_decay=0.995,
                   **kwargs):
        # Define custom training loop for strategy here
        # to use **kwargs for extra args that you might want to add
        # you can access them via x = kwargs.get("your_arg_here")

        episode_rewards = []
        total_steps = 0
        warmup = True # At the start do random actions to fill replay buffer

        for episode in range(episodes):

            if warmup: # Check if were warming up
                if total_steps > warmup_steps:
                    warmup = False

            if epsilon is not None and not warmup:
                epsilon = max(0.01, epsilon * epsilon_decay)

            state, info = self.env.reset()
            action = self.choose_action(state, epsilon, warmup=warmup)
            done = False
            episode_reward = 0


            while not done:
                next_state, reward, done, _, _ = self.env.step(action.item())
                self.memory_buffer.push((state, action, reward, next_state, done))
                state = next_state
                action = self.choose_action(state, epsilon, warmup=warmup)
                episode_reward += reward
                total_steps += 1


                if warmup:
                    # If warming up dont update NN
                    continue

                loss = self.calculate_loss(sample_batch)
                if loss is not None:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()


            episode_rewards.append(episode_reward)

            if episode % t_net_update_freq == 0:
                # Update target network every x episodes
                self.update_target_network()

        self.save_model(model_name)
