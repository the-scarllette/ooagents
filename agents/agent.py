import json
import numpy as np
from pathlib import Path
import random
from typing import List, Tuple, Type

class Agent:

    def __init__(self, actions: List[int] | Tuple[int]):
        self.actions = actions
        return

    def choose_action(self, state: np.ndarray, possible_actions: List[int]|None=None,
                      no_random: bool=False) -> int|float:
        if no_random:
            return self.actions[0]
        return random.choice(self.actions)

    def learn(self, state: np.ndarray, action: int|float, next_state: np.ndarray, reward: float,
              next_state_possible_actions: List[int]|None=None) -> None:
        return None

    @staticmethod
    def load(load_path: Path) -> 'Agent':
        with load_path.open('r') as f:
            agent_data = json.load(f)
        return Agent(agent_data['actions'])

    def save(self, save_path: Path) -> None:
        with save_path.open('w') as f:
            json.dump({'actions': self.actions}, f)
        return
