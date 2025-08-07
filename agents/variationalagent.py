import numpy as np
from typing import List, Tuple
from abc import abstractmethod
from agents.agent import Agent

class VariationalAgent(Agent):

    def __init__(self, actions: int | List[int] | Tuple[int]):
        super().__init__(actions)
        return

    # sample from a replay buffer

    # add to a replay buffer
    @abstractmethod
    def learn_representation(
            self,
            state: np.ndarray,
            action: int|float,
            reward: float,
            next_state: np.ndarray,
            skill: np.ndarray,
            terminal: bool=False,
            next_state_possible_actions: List[int]|None=None
    ) -> None:
        return

    @abstractmethod
    def learn_skill(
            self,
            skill: int,
            state: np.ndarray,
            action: int | float,
            reward: float,
            next_state: np.ndarray,
            terminal: bool = False,
            next_state_possible_actions: List[int] | None = None
    ) -> None:
        return

