# Exploration Strategies for Deep Q-Networks in CartPole-v1
This repository contains implementations and evaluations of several exploration strategies for Deep Q-Networks (DQN) in the CartPole-v1 environment. Each strategy modifies the ε-greedy exploration mechanism to improve learning stability, recovery from stagnation, and adaptability to changing dynamics.

## Setup
1. Create venv
2. run "pip install -r requirements.txt"

## Structure
1. Strategies folder stores the base model code as well as the three strategies implemented
2. Models folder stores trained models ready for evaluation
3. Test Script is used for running evaluation on models
4. Plots folder stores generated plots from evaluation script

## Models

### BaseModel - Base model without any strategies implemented
The baseline model used for evaluating the proposed exploration strategies is a Deep Q-Network (DQN) with a time-based exponential 
𝜖
ϵ-greedy decay. The agent employs a replay memory buffer of 10,000 transitions, a learning rate of 
1×10^−4, a discount factor (𝛾) of 0.8, and a batch size of 256. The exploration rate (𝜖) decays exponentially at a rate of 0.995 per episode, starting from an initial value of 0.5.

The Q-network consists of three fully connected layers:
- Input layer of dimension 4 (environment state)
- Two hidden layers with 128 ReLU units each
- Output layer of dimension 2 (available actions)

Training was performed on three pole lengths — 0.4 m, 1.1 m, and 1.8 m — for 500 episodes each. A warm-up phase of 200 random samples per pole length was used to populate the replay buffer before learning began.

This baseline configuration served as the foundation for all subsequent exploration strategy experiments and comparisons.


### Panic Bump - Model with Panic Bump strategy implemented
The Panic Bump model extends the baseline Deep Q-Network (DQN) by introducing a reward-adaptive ε-greedy exploration strategy. When the agent’s performance drops below a threshold (total reward < 120), the exploration rate ε temporarily increases by 0.08, capped at 0.5, before decaying exponentially (τ = 30,000) back toward 0.02.

This mechanism helps the agent recover from stagnation or overconfidence by boosting exploration after poor episodes. It maintains all other DQN parameters — replay buffer (10,000), batch size (256), learning rate (1e-4), discount factor (γ = 0.8), and target network updates every 50 episodes.

Training was performed on the CartPole-v1 environment with 30 pole lengths (0.4 m–1.8 m) over 200 episodes each, ensuring consistent comparison with other exploration strategies.


### Trend Based Epsilon - Model with Trend Based Adaptive Epsilon strategy implemented

The Adaptive Epsilon model extends the baseline DQN with a performance-driven ε adjustment. After each episode, the agent compares the rolling average reward of the last 25 episodes to the previous 25.

If performance improves by more than 5%, ε decays gradually (ε = ε × 0.995) to encourage exploitation.

If improvement drops below 5%, ε increases by 5% (ε = min(0.01, ε × 1.05)) to boost exploration.

Epsilon values were bounded between 0.00001 and 0.01 to balance stability and adaptability. This approach helps the agent recover from stagnation and adapt to changing environments, such as varying pole lengths in CartPole-v1.

Training was conducted on three pole lengths (0.4 m, 1.1 m, 1.8 m) for 500 episodes each, with consistent hyperparameters across experiments. An initial 200-step random warm-up phase was used for the first pole length to populate the replay buffer.


### Threshold triggered decay - Model with Threshold Triggered Decay strategy implemented
The Threshold Triggered Decay model introduces a performance-based ε adjustment mechanism to ensure that exploration is reduced only after the agent shows consistent improvement. After each episode, the agent computes the average reward over a sliding window of recent episodes (W), which expands from 50 to 200 as training progresses for greater stability.

If the average reward exceeds a dynamic threshold (T), ε decays slowly (ε = max(ε × 0.999, 0.05)), while the threshold gradually increases (T = min(T + 0.5, 30)), requiring progressively higher performance to continue exploitation. This creates a self-regulating system that balances exploration and exploitation based on learning stability.

Training began with ε = 0.9, a minimum of 0.05, and an initial threshold T = 25. The agent was trained across 30 pole lengths (0.4 m–1.8 m), running 300 episodes per length (9,000 total), promoting strong generalization.

All other hyperparameters were consistent with previous experiments: batch size = 64, γ = 0.99, learning rate = 0.001, target network updates every 200 episodes, and a 500-step warm-up for the first pole length only.