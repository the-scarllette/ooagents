import gymnasium as gym
from pathlib import Path
from typing import Type

from agents.agent import Agent

def load_agent(
        environment: gym.Env,
        load_path: Path,
        agent_type: Type[Agent]
) -> Agent:
    return agent_type.load(environment, load_path)