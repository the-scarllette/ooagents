import flax
import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax
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
                 learning_rate: float=1e-4, gamma: float=0.9, tau: float=1e-3, batch_size: int=1e3,
                 start_epsilon: float=0.1, end_epsilon: float=0.01, epsilon_scheduler: float=0.9,
                 pre_learning_steps: int=1e3, learning_frequency: int=10, target_network_update_freq: int=100):
        self.actions: List[int] = actions
        self.action_dim: int = len(actions)

        self.state_shape: Tuple[int] = state_shape

        self.gamma: float = gamma

        self.tau: float = tau

        self.batch_size: int = batch_size

        self.epsilon: float = start_epsilon
        self.end_epsilon: float = end_epsilon
        self.epsilon_scheduler: float = epsilon_scheduler
        self.start_epsilon: float = start_epsilon

        self.pre_learning_steps: int = pre_learning_steps
        self.learning_frequency: int = learning_frequency
        self.target_network_update_freq: int = target_network_update_freq
        self.training_steps: int = 0

        self.q_network: QNetwork = QNetwork(action_dim=self.action_dim,
                                            shape=network_shape)
        start_key = jax.random.PRNGKey(0)
        self.q_state = TrainState.create(
            apply_fn=self.q_network.apply,
            params=self.q_network.init(start_key),
            target_params=self.q_network.init(start_key),
            tx=optax.adam(learning_rate)
        )
        self.q_network.apply = jax.jit(self.q_network.apply)

        self.replay_buffer = ReplayBuffer(
            buffer_size,
            self.state_shape,
            self.action_dim,
            handle_timeout_termination=False
        )

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

        q_values = self.q_network.apply(self.q_state.params, state)

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
        done = 0
        if terminal:
            done = 1
        self.replay_buffer.add(state,
                               next_state,
                               np.array([action]),
                               np.array([reward]),
                               np.array([done]),
                               None
        )

        self.training_steps += 1
        if (self.training_steps <= self.pre_learning_steps) or (self.training_steps % self.learning_frequency != 0):
            return

        self.epsilon_schedule()

        transitions = self.replay_buffer.sample(self.batch_size)
        self.train_network(transitions.observations.numpy(),
                           transitions.actions.numpy(),
                           transitions.rewards.flatten().numpy(),
                           transitions.next_observations.numpy(),
                           transitions.dones.flatten().numpy()
        )

        if self.training_steps % self.target_network_update_freq == 0:
            self.q_state = self.q_state.replace(
                target_params=optax.incremental_update(self.q_state.params, self.q_state.target_params, self.tau)
            )
        return

    @staticmethod
    def load(load_path: Path) -> 'DQN':
        pass

    def save(self, save_path: Path) -> None:
        pass

    @jax.jit
    def train_network(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray,
                      terminals: np.ndarray) -> None:
        next_state_values = self.q_network.apply(self.q_state.target_params, next_states) # (batch_size, num_actions)
        max_next_state_value = jnp.max(next_state_values, axis=-1) # (batch_size, )
        target_value = rewards + ((1.0 - terminals) * self.gamma * max_next_state_value) # (batch_size, )

        def mse_loss() -> Tuple[jnp.ndarray, jnp.ndarray]:
            state_value = self.q_network.apply(self.q_state.params, states) # (batch_ize, num_actions)
            # Extracting state values for actions taken:
            state_action_value = state_value[jnp.arrange(state_value.shape[0]), actions.squeeze()] # (batch_size, )
            return ((state_action_value - target_value)**2).mean(), state_action_value

        (_, _), grads = jax.value_and_grad(mse_loss, has_aux=True)()
        self.q_state = self.q_state.apply_gradients(grads=grads)
        return
