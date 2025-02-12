from pathlib import Path
from typing import Type

from agents.agent import Agent

def load_agent(load_path: Path, agent_type: Type[Agent]) -> Agent:
    return agent_type.load(load_path)