import json
import numpy as np
from pathlib import Path
import random
from typing import List, Tuple, Type
from abc import ABC, abstractmethod

class Agent(ABC):

    def __init__(self, actions: int| List[int] | Tuple[int]):
        if type(actions) == int:
            self.actions = range(actions)
        else:
            self.actions = actions
        return

    def choose_action(self, state: np.ndarray, possible_actions: List[int]|None=None,
                      no_random: bool=False) -> int|float:
        if no_random:
            return self.actions[0]
        return random.choice(self.actions)

    @abstractmethod
    def learn(self, state: np.ndarray, action: int|float, reward: float, next_state: np.ndarray,
              terminal: bool=False, next_state_possible_actions: List[int]|None=None) -> None:
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
