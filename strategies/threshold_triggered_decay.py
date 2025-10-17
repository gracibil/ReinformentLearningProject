import os
import random
import torch
import numpy as np
from strategies.base_model import BaseDeepQModel


class AdaptiveEpsilonDeepQModel(BaseDeepQModel):
    """
    Deep Q-Learning model with adaptive epsilon strategy.

    """
    def train_loop(self, 
                   model_name,
                   episodes_per_length=300,       
                   initial_epsilon=0.9,
                   min_epsilon=0.05,
                   threshold=25,
                   t_net_update_freq=200,
                   sample_batch=64,
                   warmup_steps=500,
                   **kwargs):
        """Main training loop parameters:
        model_name: Filename to use when saving the trained model.
        episodes_per_length: Number of training episodes to run per pole length.
        initial_epsilon: Starting exploration probability.
        min_epsilon: Minimum exploration probability.
        threshold: Reward threshold used to trigger a small epsilon reduction.
        t_net_update_freq: Frequency (in episodes) to update the target network.
        sample_batch: Batch size used for sampling from replay buffer.
        warmup_steps: Number of environment steps to run with random actions before training.
        """
        #Pole length schedule and compute total episodes 300*30=9000 in this case
        pole_lengths = np.linspace(0.4, 1.8, 30).tolist()
        total_episodes = episodes_per_length * len(pole_lengths)

        #Epsilon initialized to the value given as function parameter
        epsilon = initial_epsilon
        
        #logs for later analysis and plotting
        episode_rewards = []
        epsilon_log = []
        episode_log = []

        total_steps = 0 #Counts total environment steps to determine when warmup phase ends
        warmup_done = False

        #Initialize environment with the first pole length
        length_idx = 0
        current_length = pole_lengths[length_idx]
        self.update_pole_length(current_length)

        #Making sure the models directory exists
        if not os.path.exists('../models'):
            os.makedirs('../models')

        #Parameters for the adaptive window, used to compute average reward
        base_window = 50
        max_window = 200
        window_step = 30
        window_increase_every = 300  #every 300 episodes the window size is increased by window step
        current_window = base_window
        max_threshold = 30
 
        #Main episode loop
        for episode in range(total_episodes):
            #Change pole length every "episodes_per_length" episodes (e.g., every 300 episodes)
            if episode > 0 and episode % episodes_per_length == 0:
                length_idx = (length_idx + 1) % len(pole_lengths)
                current_length = pole_lengths[length_idx]
                self.update_pole_length(current_length)
                print(f"\n--> Pole length changed to {current_length:.2f} at episode {episode}")

            state, info = self.env.reset()
            done = False
            episode_reward = 0
            
            #Main loop for one episode (actions,transitions)
            while not done:
                #if still warming up continue to sample random actions
                if not warmup_done:
                    action = self.env.action_space.sample()
                else:
                    #Epsilon greedy: random action selection
                    if random.random() < epsilon:
                        action = self.env.action_space.sample()
                    else:
                        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                        with torch.no_grad():
                            action = self.QNetwork(state_tensor).argmax().item()
                
                #Execute the chosen action in the environment and observe next state, reward, and done flags
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                is_done = terminated or truncated

                #Transition stored in replay buffer
                self.memory_buffer.push((state, action, reward, next_state, is_done))
                
                state = next_state
                episode_reward += reward
                total_steps += 1

                #warmup phase passed track, to sample random actions in the beginning
                if not warmup_done and total_steps >= warmup_steps:
                    warmup_done = True

                #After warmup and enough samples in the buffer: network update
                if warmup_done and len(self.memory_buffer.buffer) >= sample_batch:
                    loss = self.calculate_loss(sample_batch)
                    if loss is not None:
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                done = is_done
            
            episode_rewards.append(episode_reward) 
            episode_log.append(episode)
            epsilon_log.append(epsilon)

            #Reward adaptive epsilon strategy
            if warmup_done:
                #Average reward over recent "current_window" episodes to decide epsilon adaptation
                avg_display = np.mean(episode_rewards[-current_window:])

                #If the average rewards computed exceeds the threshold, reduce epsilon
                if avg_display >= threshold:
                    epsilon *= 0.999
                    #Making the threshold harder to reach over time, but also with a top limit which is chosen to be 30 (max_threshold)
                    threshold = min(threshold + 0.5, max_threshold)
                    epsilon = max(min_epsilon, epsilon)
 
                #Gradually increase the averaging window to make performance evaluation more stable over time
                if (episode+1) % window_increase_every == 0:
                    current_window = min(current_window + window_step, max_window)

            #Print progress every 10 episodes
            if episode % 10 == 0:
                if warmup_done:
                    print(f"\rEp {episode+1}/{total_episodes} | Reward: {episode_reward:.1f} | Avg{current_window}: {avg_display:.1f} | Eps: {epsilon:.3f} | Pole: {current_length:.2f}", end="", flush=True)
                else:
                    print(f"\rEp {episode+1}/{total_episodes} | Reward: {episode_reward:.1f} | Eps: {epsilon:.3f} | Pole: {current_length:.2f}", end="", flush=True)

            #Update target network on schedule
            if episode % t_net_update_freq == 0:
                self.update_target_network()

        # #Save epsilon and reward logs for external analysis
        # epsilon_df = pd.DataFrame({
        #     "Episode": episode_log,
        #     "Epsilon": epsilon_log,
        #     "Reward": episode_rewards
        # })
        # epsilon_df.to_csv("epsilon_progress.csv", index=False)

        # # Plot for epsilon decay and reward progression
        # fig, ax1 = plt.subplots(figsize=(8,5))

        # color = 'tab:blue'
        # ax1.set_xlabel('Episode')
        # ax1.set_ylabel('Epsilon', color=color)
        # ax1.plot(episode_log, epsilon_log, color=color, linewidth=2, label='Epsilon')
        # ax1.tick_params(axis='y', labelcolor=color)
        # ax1.grid(True, alpha=0.3)

        # # Add second y-axis for reward
        # ax2 = ax1.twinx()  
        # color = 'tab:orange'
        # ax2.set_ylabel('Reward', color=color)
        # ax2.plot(episode_log, episode_rewards, color=color, alpha=0.6, label='Reward')
        # ax2.tick_params(axis='y', labelcolor=color)

        # plt.title("Epsilon Decay and Episode Reward Progression")
        # fig.tight_layout()
        # plt.savefig("epsilon_reward_dual_plot.png")
        # plt.show()
 
        #Save trained model
        self.save_model(model_name)
        print(f"\nTraining finished. Model saved as ./models/{model_name}.pth")


if __name__ == "__main__":
    STATE_DIM = 4
    ACTION_DIM = 2

    model = AdaptiveEpsilonDeepQModel(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        memory_buffer_size=50000,
        learning_rate=0.001,
        discount_factor=0.99
    )

    model.train_loop(
        model_name="reward_adaptive_30lengths",
        episodes_per_length=300,  
        initial_epsilon=0.9,
        min_epsilon=0.05,
        threshold=25,
        warmup_steps=500
    )
