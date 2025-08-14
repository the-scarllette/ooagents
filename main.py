import gymnasium as gym
import jax
from pathlib import Path
import os

from agents.diayn import DIAYN
from agents.dqn import DQN
from agents.sac import SAC
from training.agent_training import train_agent
from utils.load_agent import load_agent

if __name__ == '__main__':

    environment = gym.make("Ant-v5")

    diayn_agent = DIAYN(
        environment,
        True,
        num_skills=3,
        discrete_skills=True
    )

    diayn_agent.train_skills(
        environment,
        100_000
    )

    diayn_agent.save(
        Path("trained-agents/diayn-test")
    )

    exit()

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
