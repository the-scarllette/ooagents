import gymnasium as gym
import jax
from pathlib import Path
import os

from agents.dqn import DQN
from agents.sac import SAC
from training.agent_training import train_agent
from utils.load_agent import load_agent

if __name__ == '__main__':

    environment = gym.make('CartPole-v1')

    dqn_agent = DQN(
        environment,
        [128, 84],
        10000
    )

    print("Training Agent")
    train_agent(
        environment,
        dqn_agent,
        50_000,
        0,
        0,
        progress_bar=True
    )

    environment = gym.make("CartPole-v1")
    episode_over = False
    state, _ = environment.reset()
    total_reward_pre_load = 0
    while not episode_over:
        action = dqn_agent.choose_action(state)  # agent policy that uses the observation and info
        state, reward, terminated, truncated, _ = environment.step(action)
        episode_over = terminated or truncated
        total_reward_pre_load += reward

    dqn_agent.save(Path('trained-agents/dqn_test'))
    new_agent = load_agent(environment, Path('trained-agents/dqn_test'), DQN)

    environment = gym.make("CartPole-v1")
    episode_over = False
    state, _ = environment.reset()
    total_reward_post_load = 0
    while not episode_over:
        action = new_agent.choose_action(state)  # agent policy that uses the observation and info
        state, reward, terminated, truncated, _ = environment.step(action)
        episode_over = terminated or truncated
        total_reward_post_load += reward

    print("Reward pre load: " + str(total_reward_pre_load))
    print("Reward post load: " + str(total_reward_post_load))
    exit()

    environment = gym.make("CartPole-v1")

    print("Creating SAC Agent")
    sac_agent = SAC(
        environment,
        False,
        network_shape=[64, 64]
    )

    print("Running initial environment")
    environment = gym.make("CartPole-v1")
    episode_over = False
    state, _ = environment.reset()
    total_reward_pre_training = 0
    while not episode_over:
        action = sac_agent.choose_action(state)  # agent policy that uses the observation and info
        state, reward, terminated, truncated, _ = environment.step(action)
        episode_over = terminated or truncated
        total_reward_pre_training += reward

    print("Training Agent")
    train_agent(
        environment,
        sac_agent,
        100_000,
        0,
        0,
        progress_bar=True
    )

    print("Running final environment")
    episode_over = False
    environment = gym.make("CartPole-v1", render_mode="human")
    state, _ = environment.reset()
    total_reward = 0
    while not episode_over:
        action = sac_agent.choose_action(state)  # agent policy that uses the observation and info
        state, reward, terminated, truncated, _ = environment.step(action)
        episode_over = terminated or truncated
        total_reward += reward

    print("Reward pre training: " + str(total_reward_pre_training))
    print("Reward post training: " + str(total_reward))
    exit()
