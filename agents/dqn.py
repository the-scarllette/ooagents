import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import random
from stable_baselines3.common.buffers import ReplayBuffer
from typing import List, Tuple

from agent import Agent

class QNetwork(nn.Module):
    action_dim: int
    shape: List[int]

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for size in self.shape:
            x = nn.Dense(size)(x)
            x = nn.relu(x)

        x = nn.Dense(self.action_dim)(x)
        return x

class DQN(Agent):

    def __init__(self, actions: List[int], state_shape: Tuple[int],
                 network_shape: List[int], buffer_size: int,
                 start_epsilon: float=0.1, end_epsilon: float=0.01, epsilon_scheduler: float=0.9):
        self.actions: List[int] = actions
        self.action_dim: int = len(actions)

        self.state_shape: Tuple[int] = state_shape

        self.epsilon: float = start_epsilon
        self.end_epsilon: float = end_epsilon
        self.epsilon_scheduler: float = epsilon_scheduler
        self.start_epsilon: float = start_epsilon

        self.q_network: QNetwork = QNetwork(action_dim=self.action_dim,
                                            shape=network_shape)
        self.q_network.apply = jax.jit(self.q_network.apply)

        self.replay_buffer = ReplayBuffer(buffer_size, self.state_shape, self.action_dim,
                                          handle_timeout_termination=False,)
        return

    def action_mask(self, action_values: jnp.array, possible_actions: List[int]) -> jnp.ndarray:
        mask = jnp.ones(self.q_network.action_dim, dtype=int)
        mask[possible_actions] = 0

        action_values[mask] = -np.inf
        return action_values

    def choose_action(self, state: np.ndarray, possible_actions: List[int]|None=None,
                      no_random: bool=False) -> int|float:
        if possible_actions is None:
            possible_actions = self.actions

        if (not no_random) and random.uniform(0, 1) < self.epsilon:
            return random.choice(possible_actions)

        q_values = self.q_network.apply(state)

        if possible_actions is not None:
            q_values = self.action_mask(q_values, possible_actions)

        actions = q_values.argmax(axis=-1)
        actions = jax.device_get(actions)
        return actions

    def epsilon_schedule(self) -> None:
        self.epsilon = max(self.epsilon_scheduler * self.epsilon, self.end_epsilon)
        return

    def learn(self, state: np.ndarray, action: int|float, reward: float, next_state: np.ndarray,
              terminal: bool=False, next_state_possible_actions: List[int]|None=None) -> None:
        self.replay_buffer.add(state, next_state, action, reward, done, None)
        pass

    @staticmethod
    def load(load_path: Path) -> 'DQN':
        pass

    def save(self, save_path: Path) -> None:
        pass
