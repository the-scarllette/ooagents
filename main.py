from pathlib import Path

from agents import Agent
from utils import load_agent

if __name__ == '__main__':

    test_agent = Agent([0, 1, 2])
    agent_path = Path("test_agent.json")
    test_agent.save(agent_path)

    test_agent = load_agent(agent_path, Agent)
    print(test_agent.actions)

