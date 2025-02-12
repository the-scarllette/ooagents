import gymnasium as gym
from pathlib import Path

from agents.dqn import DQN
from training.agent_training import train_agent

if __name__ == '__main__':

    environment = gym.make("CartPole-v1")
    print(environment.action_space.n)

    #dqn_agent = DQN(actions=buffer_size=10_000)
