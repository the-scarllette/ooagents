import numpy as np
from typing import List, Tuple

from agents.agent import Agent

class VariationalAgent(Agent):

    def __init__(self, actions: int | List[int] | Tuple[int]):
        super().__init__(actions)
        return

    def choose_action(
            self,
            state: np.ndarray,
            possible_actions: List[int] | None = None,
            no_random: bool = False
    ) -> int | float:
        pass

    def learn(
            self,
            state: np.ndarray,
            action: int|float,
            reward: float,
            next_state: np.ndarray,
            terminal: bool=False,
            next_state_possible_actions: List[int]|None=None
    ) -> None:
        pass
