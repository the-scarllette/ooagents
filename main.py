import gymnasium as gym
import jax
from pathlib import Path
import os

from agents.dqn import DQN
from training.agent_training import train_agent

if __name__ == '__main__':

    environment = gym.make("CartPole-v1")

    print("Creating DQN Agent")
    dqn_agent = DQN(environment,
                    network_shape=[128, 64],
                    buffer_size=10_000,
                    batch_size=128,
                    target_network_update_freq=500,
                    pre_learning_steps=10_000
    )

    print("Running initial environment")
    environment = gym.make("CartPole-v1")
    episode_over = False
    state, _ = environment.reset()
    total_reward_pre_training = 0
    while not episode_over:
        action = dqn_agent.choose_action(state, no_random=True)  # agent policy that uses the observation and info
        state, reward, terminated, truncated, _ = environment.step(action)
        episode_over = terminated or truncated
        total_reward_pre_training += reward

    environment = gym.make("CartPole-v1")

    print("Training Agent")
    train_agent(environment, dqn_agent,
                500_000, 0, 0,
                progress_bar=True)

    print("Running final environment")
    episode_over = False
    environment = gym.make("CartPole-v1", render_mode="human")
    state, _ = environment.reset()
    total_reward = 0
    while not episode_over:
        action = dqn_agent.choose_action(state, no_random=True)  # agent policy that uses the observation and info
        state, reward, terminated, truncated, _ = environment.step(action)
        episode_over = terminated or truncated
        total_reward += reward

    print("Reward pre training: " + str(total_reward_pre_training))
    print("Reward post training: " + str(total_reward))
    exit()
