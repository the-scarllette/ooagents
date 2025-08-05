import numpy as np
from typing import List, Tuple

from agents.variationalagent import VariationalAgent

class DIAYN(VariationalAgent):

    def __init__(
            self,
            actions: int | List[int] | Tuple[int],
            policy_shape: List[int],
            discriminator_shape: List[int],
            buffer_size: int,
            learning_rate: float=3e-4,
            discrete_skills: bool = True,
            num_skills: None | int = 3,
    ):
        super().__init__(actions)
        return
