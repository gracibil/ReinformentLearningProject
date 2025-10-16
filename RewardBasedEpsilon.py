from base_model import BaseDeepQModel
import numpy as np

class RewardBasedEpsilonModel(BaseDeepQModel):
    def adjust_epsilon(self, epsilon, episode_rewards, epsilon_decay, epsilon_increase = 1.05):
        episode_count = len(episode_rewards)
        if episode_count > 50:
            old_avg_start = episode_count - 50
            new_avg_start = episode_count - 25
            old_avg = sum(episode_rewards[old_avg_start:new_avg_start]) / 25
            new_avg = sum(episode_rewards[new_avg_start:episode_count]) / 25
            trend = new_avg - old_avg
            percent_increase = (trend / abs(old_avg)) if old_avg != 0 else 0

            if percent_increase > 0.05:  # If the average reward is increasing exploit the current policy
                epsilon = epsilon * epsilon_decay
            else:  # Otherwise if the average reward is not increasing enough aggresively explore more
                epsilon = min(0.01, epsilon * epsilon_increase)


            return max(0.00001, epsilon)

        else:
            return epsilon





    def train_loop(self, episodes, epsilon=None,
                   t_net_update_freq=25, sample_batch=64,
                   warmup_steps=1000, epsilon_decay=0.975, warmup=True,
                   **kwargs):

        # Define custom training loop for strategy here
        # to use **kwargs for extra args that you might want to add
        # you can access them via x = kwargs.get("your_arg_here")

        episode_rewards = []
        total_steps = 0
        warmup = warmup # At the start do random actions to fill replay buffer


        for episode in range(episodes):

            if warmup: # Check if were warming up
                if total_steps > warmup_steps:
                    warmup = False

            if epsilon is not None and not warmup:
                # Custom epsilon decay based on average reward

                epsilon = self.adjust_epsilon(epsilon, episode_rewards, epsilon_decay)

            state, info = self.env.reset()
            action = self.choose_action(state, epsilon, warmup=warmup)
            done = False
            episode_reward = 0


            while not done:
                next_state, reward, done, _, _ = self.env.step(action)
                # Custom reward shaping

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

                # Condition to break if the model is doing well to prevent infinite loops
                if episode_reward >= 2000:
                    break


            episode_rewards.append(episode_reward)
            last_50_rewards = episode_rewards[-50:]
            avg_last_50 = sum(last_50_rewards) / len(last_50_rewards)
            avg_last_25 = sum(last_50_rewards[-25:]) / len(last_50_rewards[-25:])
            avg_50_25 = sum(last_50_rewards[0:25]) / len(last_50_rewards[0:25])

            print(f"\rEpisode: {episode}, Reward: {episode_reward}, Epsilon : {epsilon}, avg_rewards_last_25 : {avg_last_25} avg_rewards_old_25 : {avg_50_25}", end="", flush=True)

            if episode % t_net_update_freq == 0:
                # Update target network every x episodes
                self.update_target_network()

            if sum(episode_rewards[-50:])/50 >= 2000:
                # Condition to break if the model is doing well to stop training early
                print("\nEnvironment solved in", episode, "episodes!")
                self.update_target_network()
                break



if __name__ == "__main__":
    pole_lengths = np.linspace(0.4, 1.8, 3)
    model = RewardBasedEpsilonModel(state_dim=4, action_dim=2, memory_buffer_size=10000, learning_rate=0.0001, discount_factor=0.8)
    first_run = True
    for length in pole_lengths:

        print(' ----------------------- ')
        print('Training for pole length:', length)
        print(' ----------------------- ')
        model.update_pole_length(length)
        if first_run:
            model.train_loop(500, epsilon=0.2, warmup_steps=200, sample_batch=256, warmup=True)
            first_run = False
        else:
            model.train_loop(500, epsilon=0.01, warmup_steps=0, sample_batch=256, warmup=False)
        print('\n')

    model.save_model('reward_based_epsilon_model')