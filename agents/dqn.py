import numpy as np
from pathlib import Path
from typing import List

from agent import Agent

class DQN(Agent):

    def __init__(self, state_size, action_size):
        return

    def choose_action(self, state: np.ndarray, possible_actions: List[int]|None=None,
                      no_random: bool=False) -> int|float:
        pass

    def learn(self, state: np.ndarray, action: int|float, reward: float, next_state: np.ndarray,
              terminal: bool=False, next_state_possible_actions: List[int]|None=None) -> None:
        pass

    @staticmethod
    def load(load_path: Path) -> 'DQN':
        pass

    def save(self, save_path: Path) -> None:
        pass
