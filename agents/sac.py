import flax
import flax.linen as nn
from flax.training import orbax_utils
from flax.training.train_state import TrainState
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax
from pathlib import Path
from jax import Array
from stable_baselines3.common.buffers import ReplayBuffer
from typing import List, Tuple

from agents.agent import Agent

class QNetwork(nn.Module):
    action_dim: int
    shape: List[int]

    @nn.compact
    def __call__(
            self,
            x: jnp.ndarray,
    ) -> jnp.ndarray:
        for size in self.shape:
            x = nn.Dense(size)(x)
            x = nn.relu(x)

        x = nn.Dense(self.action_dim)(x)
        return x

class PolicyNetwork(nn.Module):
    action_dim: int
    shape: List[int]

    LOG_STD_MIN: float=-5
    LOG_STD_RANGE: float = 7 # LOG_MAX - LOG_MIN

    @nn.compact
    def __call__(
            self,
            x: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        for size in self.shape:
            x = nn.Dense(size)(x)
            x = nn.relu(x)

        mean = nn.Dense(self.action_dim)(x)

        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * self.LOG_STD_RANGE * (log_std + 1)
        return mean, log_std

class SoftQNetwork(nn.Module):
    shape: List[int]
    action_dim: int = 1

    @nn.compact
    def __call__(
            self,
            observations: jnp.ndarray,
            actions: jnp.ndarray,
    ) -> jnp.array:
        x = jnp.concatenate([observations, actions], axis=1)
        for size in self.shape:
            x = nn.Dense(size)(x)
            x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

class ValueNetwork(nn.Module):
    shape: List[int]

    @nn.compact
    def __call__(
            self,
            observations: jnp.ndarray
    ) -> jnp.ndarray:
        for size in self.shape:
            observations = nn.Dense(size)(observations)
            observations = nn.relu(observations)
        observations = nn.Dense(1)(observations)
        return observations

class TrainState(TrainState):
    target_params: flax.core.FrozenDict

class SAC(Agent):

    def __init__(
            self,
            environment: gym.Env,
            continuous_actions: bool,
            network_shape: None|List[int]=None,
            buffer_size: int=1_000_000,
            reward_scale: float=0.1,
            learning_rate: float=3e-4,
            gamma: float=0.99,
            tau: float=1.0,
            minibatch_size: int=256,
            pre_learning_steps: int=5e3,
            learning_frequency: int=500,
            output_loss_freq: int=1e4,
            target_network_update_freq: int=500,
            state_shape: None|Tuple[int, ...]=None,
            observation_sample: None|jnp.ndarray|np.ndarray=None,
    ):
        self.network_shape: None|List[int] = network_shape
        if network_shape is None:
            self.network_shape = [256, 256]
        self.continuous_actions = continuous_actions

        self.state_shape: Tuple[int, ...] = environment.observation_space.shape
        if state_shape is not None:
            self.state_shape = state_shape
        self.buffer_size: int = buffer_size
        self.minibatch_size: int = minibatch_size
        self.learning_rate: float = learning_rate
        self.gamma: float = gamma
        self.reward_scale: float = reward_scale
        self.tau: float = tau

        self.pre_learning_steps: int = pre_learning_steps
        self.learning_frequency: int = learning_frequency
        self.output_loss_freq: int = output_loss_freq
        self.target_network_update_freq: int = target_network_update_freq

        self.policy_network: PolicyNetwork|QNetwork
        self.q_network_1: SoftQNetwork|QNetwork
        self.q_network_2: SoftQNetwork|QNetwork
        if self.continuous_actions:
            self.action_shape = environment.action_space.shape
            self.policy_network = PolicyNetwork(
                action_dim=self.action_shape[0],
                shape=self.network_shape
            )
            self.q_network_1: SoftQNetwork = SoftQNetwork(
                shape=self.network_shape
            )
            self.q_network_2: SoftQNetwork = SoftQNetwork(
                shape=self.network_shape
            )
        else:
            self.action_shape = (environment.action_space.n,)
            self.policy_network = QNetwork(
                action_dim=self.action_shape[0],
                shape=self.network_shape
            )
            self.q_network_1 = QNetwork(
                action_dim=self.action_shape[0],
                shape=self.network_shape
            )
            self.q_network_2 = QNetwork(
                action_dim=self.action_shape[0],
                shape=self.network_shape
            )

        self.value_network: ValueNetwork = ValueNetwork(
            shape=self.network_shape
        )

        start_key = jax.random.PRNGKey(0)
        obs, _ = environment.reset()
        if observation_sample is not None:
            obs = observation_sample
        self.policy: TrainState = TrainState.create(
            apply_fn=self.policy_network.apply,
            params=self.policy_network.init(start_key, obs),
            target_params=self.policy_network.init(start_key, obs),
            tx=optax.adam(learning_rate=self.learning_rate)
        )
        self.policy_network.apply = jax.jit(self.policy_network.apply)

        if self.continuous_actions:
            self.q_1: TrainState = TrainState.create(
                apply_fn=self.q_network_1.apply,
                params=self.q_network_1.init(start_key, jnp.array([obs]), jnp.array([environment.action_space.sample()])),
                target_params=self.q_network_1.init(start_key, jnp.array([obs]), jnp.array([environment.action_space.sample()])),
                tx=optax.adam(learning_rate=self.learning_rate)
            )
            self.q_network_1.apply = jax.jit(self.q_network_1.apply)
            self.q_2: TrainState = TrainState.create(
                apply_fn=self.q_network_2.apply,
                params=self.q_network_2.init(start_key, jnp.array([obs]), jnp.array([environment.action_space.sample()])),
                target_params=self.q_network_2.init(start_key, jnp.array([obs]), jnp.array([environment.action_space.sample()])),
                tx=optax.adam(learning_rate=self.learning_rate)
            )
            self.q_network_2.apply = jax.jit(self.q_network_2.apply)
        else:
            self.q_1: TrainState = TrainState.create(
                apply_fn=self.q_network_1.apply,
                params=self.q_network_1.init(start_key, obs),
                target_params=self.q_network_1.init(start_key, obs),
                tx=optax.adam(learning_rate=self.learning_rate)
            )
            self.q_network_1.apply = jax.jit(self.q_network_1.apply)
            self.q_2: TrainState = TrainState.create(
                apply_fn=self.q_network_2.apply,
                params=self.q_network_2.init(start_key, obs),
                target_params=self.q_network_2.init(start_key, obs),
                tx=optax.adam(learning_rate=self.learning_rate)
            )
            self.q_network_2.apply = jax.jit(self.q_network_2.apply)

        self.value: TrainState = TrainState.create(
            apply_fn=self.value_network.apply,
            params=self.value_network.init(start_key, obs),
            target_params=self.value_network.init(start_key, obs),
            tx=optax.adam(learning_rate=self.learning_rate)
        )
        self.value_network.apply = jax.jit(self.value_network.apply)

        self.replay_buffer: ReplayBuffer = ReplayBuffer(
            buffer_size,
            environment.observation_space,
            environment.action_space,
            "cpu",
            handle_timeout_termination=False
        )

        self.training_steps: int = 0
        return

    def choose_action(
            self,
            state: np.ndarray,
            possible_actions: List[int]|None=None,
            no_random: bool=False
    ) -> np.ndarray:
        if self.continuous_actions:
            mean, log_std = self.policy_network.apply(
                self.policy.params,
                state
            )
            std = np.array(jnp.exp(log_std))

            action, _ = self.sample_action(mean, std)
            return np.array(action)

        logits = self.policy_network.apply(
            self.policy.params,
            state
        )
        action, _ = self.sample_discrete_action(logits)
        return int(action)

    def learn(
            self,
            state: np.ndarray,
            action: int|float,
            reward: float,
            next_state: np.ndarray,
            terminal: bool=False,
            next_state_possible_actions: List[int]|None=None
    ) -> None:

        def train_policy(
                states: np.ndarray
        ) -> Tuple[jnp.ndarray, TrainState]:
            def policy_loss_fn(params) -> Tuple[jnp.ndarray, jnp.ndarray]:
                if self.continuous_actions:
                    mean, log_std = self.policy_network.apply(
                        self.policy.target_params,
                        states
                    )
                    std = np.array(jnp.exp(log_std))
                    new_actions, log_probs = self.sample_action(mean, std)

                    q_1_values = self.q_network_1.apply(self.q_1.target_params, states, new_actions)
                    q_2_values = self.q_network_2.apply(self.q_2.target_params, states, new_actions)
                    predicted_q_value = jax.lax.min(q_1_values, q_2_values)
                else:
                    logits = self.policy_network.apply(
                        self.policy.target_params,
                        states
                    )
                    new_actions, log_probs = self.sample_discrete_action(logits)

                    q_1_values = self.q_network_1.apply(self.q_1.target_params, states)
                    q_2_values = self.q_network_2.apply(self.q_2.target_params, states)
                    q_1_values = q_1_values[:, new_actions]
                    q_2_values = q_2_values[:, new_actions]
                    predicted_q_value = jax.lax.min(q_1_values, q_2_values)

                loss = jnp.mean(log_probs - predicted_q_value)
                l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))

                if self.continuous_actions:
                    model_output = jnp.concatenate([mean, log_std], axis=1)
                else:
                    model_output = log_probs

                return loss + l2_loss, model_output

            (p_loss, _), grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(self.policy.params)
            new_policy_state = self.policy.apply_gradients(grads=grads)
            return p_loss, new_policy_state

        if self.continuous_actions:
            def train_q(
                    states: np.ndarray,
                    actions: np.ndarray,
                    rewards: np.ndarray,
                    next_states: np.ndarray,
                    terminals: np.ndarray,
            ) -> Tuple[jnp.array, TrainState, TrainState]:
                target_q = self.value_network.apply(self.value.target_params, next_states)
                target_q = (1 - terminals) * target_q
                target_q = rewards * self.reward_scale + self.gamma * target_q
                target_q = jax.lax.stop_gradient(target_q)

                def q1_loss_fn(params) -> Tuple[jnp.ndarray, jnp.ndarray]:
                    q1_value = self.q_network_1.apply(params, states, actions)
                    q2_value = self.q_network_2.apply(self.q_2.target_params, states, actions)

                    q1_td = target_q - q1_value
                    q2_td = target_q - q2_value

                    q1_loss = 0.5 * jnp.mean(jnp.square(q1_td))
                    q2_loss = 0.5 * jnp.mean(jnp.square(q2_td))
                    return q1_loss + q2_loss, q1_value

                def q2_loss_fn(params) -> Tuple[jnp.ndarray, jnp.ndarray]:
                    q1_value = self.q_network_1.apply(self.q_1.target_params, states, actions)
                    q2_value = self.q_network_2.apply(params, states, actions)

                    q1_td = target_q - q1_value
                    q2_td = target_q - q2_value

                    q1_loss = 0.5 * jnp.mean(jnp.square(q1_td))
                    q2_loss = 0.5 * jnp.mean(jnp.square(q2_td))
                    return q1_loss + q2_loss, q2_value

                (_, _), grads_1 = jax.value_and_grad(q1_loss_fn, has_aux=True)(self.q_1.params)
                (total_q_loss, _), grads_2 = jax.value_and_grad(q2_loss_fn, has_aux=True)(self.q_2.params)

                q1_new_state = self.q_1.apply_gradients(grads=grads_1)
                q2_new_state = self.q_2.apply_gradients(grads=grads_2)

                return total_q_loss, q1_new_state, q2_new_state
        else:
            def train_q(
                    states: np.ndarray,
                    actions: np.ndarray,
                    rewards: np.ndarray,
                    next_states: np.ndarray,
                    terminals: np.ndarray,
            ) -> Tuple[jnp.array, TrainState, TrainState]:
                target_q = self.value_network.apply(self.value.target_params, next_states)
                target_q = (1 - terminals) * target_q
                target_q = rewards * self.reward_scale + self.gamma * target_q
                target_q = jax.lax.stop_gradient(target_q)

                def q1_loss_fn(params) -> Tuple[jnp.ndarray, jnp.ndarray]:
                    q1_value = self.q_network_1.apply(params, states)
                    q1_value = q1_value[np.arange(self.minibatch_size), actions]
                    q2_value = self.q_network_2.apply(self.q_2.target_params, states)
                    q2_value = q2_value[np.arange(self.minibatch_size), actions]

                    q1_td = target_q - q1_value
                    q2_td = target_q - q2_value

                    q1_loss = 0.5 * jnp.mean(jnp.square(q1_td))
                    q2_loss = 0.5 * jnp.mean(jnp.square(q2_td))
                    return q1_loss + q2_loss, q1_value

                def q2_loss_fn(params) -> Tuple[jnp.ndarray, jnp.ndarray]:
                    q1_value = self.q_network_1.apply(self.q_1.target_params, states)
                    q1_value = q1_value[np.arange(self.minibatch_size), actions]
                    q2_value = self.q_network_2.apply(params, states)
                    q2_value = q2_value[np.arange(self.minibatch_size), actions]

                    q1_td = target_q - q1_value
                    q2_td = target_q - q2_value

                    q1_loss = 0.5 * jnp.mean(jnp.square(q1_td))
                    q2_loss = 0.5 * jnp.mean(jnp.square(q2_td))
                    return q1_loss + q2_loss, q2_value

                (_, _), grads_1 = jax.value_and_grad(q1_loss_fn, has_aux=True)(self.q_1.params)
                (total_q_loss, _), grads_2 = jax.value_and_grad(q2_loss_fn, has_aux=True)(self.q_2.params)

                q1_new_state = self.q_1.apply_gradients(grads=grads_1)
                q2_new_state = self.q_2.apply_gradients(grads=grads_2)

                return total_q_loss, q1_new_state, q2_new_state

        def train_v(
                states: np.ndarray
        ) -> Tuple[jnp.array, TrainState]:
            if self.continuous_actions:
                mean, log_std = self.policy_network.apply(
                    self.policy.target_params,
                    states
                )
                std = np.array(jnp.exp(log_std))

                new_actions, log_probs = self.sample_action(mean, std)
                q1_value = self.q_network_1.apply(self.q_1.target_params, states, new_actions)
                q2_value = self.q_network_2.apply(self.q_2.target_params, states, new_actions)
            else:
                logits = self.policy_network.apply(
                    self.policy.target_params,
                    states
                )
                new_actions, log_probs = self.sample_discrete_action(logits)

                q1_value = self.q_network_1.apply(self.q_1.target_params, states)
                q1_value = q1_value[np.arange(self.minibatch_size), new_actions]
                q2_value = self.q_network_2.apply(self.q_2.target_params, states)
                q2_value = q2_value[np.arange(self.minibatch_size), new_actions]

            q_min = jax.lax.min(q1_value, q2_value)
            target_value = q_min - log_probs
            target_value = jax.lax.stop_gradient(target_value)

            def v_loss_fn(params) -> Tuple[jnp.ndarray, jnp.ndarray]:
                predicted_value = self.value_network.apply(params, states)

                loss = 0.5 * jnp.square(predicted_value - target_value)
                loss = jnp.mean(loss)
                return loss, predicted_value

            (v_loss, _), grads = jax.value_and_grad(v_loss_fn, has_aux=True)(self.value.params)
            v_new_state = self.value.apply_gradients(grads=grads)
            return v_loss, v_new_state

        done = 0
        if terminal:
            done = 1
        self.replay_buffer.add(
            state,
            next_state,
            np.array([action]),
            np.array([reward]),
            np.array([done]),
            None
        )

        self.training_steps += 1
        if (self.training_steps <= self.pre_learning_steps) or (self.training_steps % self.learning_frequency != 0):
            return

        transitions = self.replay_buffer.sample(self.minibatch_size)

        value_loss, self.value = train_v(transitions.observations.numpy())

        q_loss, self.q_1, self.q_2 = train_q(
            transitions.observations.numpy(),
            transitions.actions.numpy(),
            transitions.rewards.flatten().numpy(),
            transitions.next_observations.numpy(),
            transitions.dones.flatten().numpy()
        )

        policy_loss, self.policy = train_policy(transitions.observations.numpy())

        if (self.output_loss_freq > 0) and (self.training_steps % self.output_loss_freq == 0):
            print("Value Loss: " + str(jax.device_get(value_loss)))
            print("Q Loss: " + str(jax.device_get(q_loss)))
            print("Policy Loss: " + str(jax.device_get(policy_loss)))

        if self.training_steps % self.target_network_update_freq == 0:
            self.policy = self.update_network(self.policy)
            self.q_1 = self.update_network(self.q_1)
            self.q_2 = self.update_network(self.q_2)
            self.value = self.update_network(self.value)

        return

    @staticmethod
    def load(environment: gym.Env, load_path: Path) -> 'SAC':
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        raw_load = orbax_checkpointer.restore(load_path.resolve())

        agent = SAC(
            environment,
            raw_load['agent_details']['continuous_actions'],
            raw_load['agent_details']['network_shape'],
            raw_load['agent_details']['buffer_size'],
            raw_load['agent_details']['reward_scale'],
            raw_load['agent_details']['learning_rate'],
            raw_load['agent_details']['gamma'],
            raw_load['agent_details']['minibatch_size'],
            raw_load['agent_details']['pre_learning_steps'],
            raw_load['agent_details']['learning_frequency'],
            raw_load['agent_details']['output_loss_freq'],
        )

        agent.policy = agent.policy.replace(
            params=raw_load['policy']['params'],
            target_params=raw_load['policy']['target_params'],
        )
        agent.q_1 = agent.q_1.replace(
            params=raw_load['q_1']['params'],
            target_params=raw_load['q_1']['target_params'],
        )
        agent.q_2 = agent.q_2.replace(
            params=raw_load['q_2']['params'],
            target_params=raw_load['q_2']['target_params'],
        )
        agent.value = agent.value_replace(
            params=raw_load['value']['params'],
            target_params=raw_load['value']['target_params'],
        )

        return agent

    @staticmethod
    def sample_action(
            mean: np.ndarray,
            std: np.ndarray
    ) -> Tuple[Array, Array]:
        action = np.random.normal(mean, std)
        action = jnp.tanh(action)

        log_prob = jax.scipy.stats.norm.logpdf(action, mean, std)
        log_prob -= jnp.log((1-jnp.pow(action, 2)) + 1e-6)
        return action, log_prob

    @staticmethod
    def sample_discrete_action(
            logits: jnp.ndarray
    ) -> Tuple[Array, Array]:
        action = jax.random.categorical(jax.random.PRNGKey(0), logits)
        if len(logits.shape) <= 1:
            log_prob = logits[action]
        else:
            log_prob = logits[np.arange(logits.shape[0]), action]
        return action, log_prob

    def save(self, save_path: Path) -> None:
        ckpt = {
            'policy': {
                'params': self.policy.params,
                'target_params': self.policy.target_params,
            },
            'q_1': {
                'params': self.q_1.params,
                'target_params': self.q_1.target_params,
            },
            'q_2': {
                'params': self.q_2.params,
                'target_params': self.q_2.target_params,
            },
            'value': {
                'params': self.value.params,
                'target_params': self.value.target_params,
            },
            'agent_details':
                {
                    'continuous_actions': self.continuous_actions,
                    'network_shape': self.network_shape,
                    'buffer_size': self.buffer_size,
                    'reward_scale': self.reward_scale,
                    'learning_rate': self.learning_rate,
                    'gamma': self.gamma,
                    'minibatch_size': self.minibatch_size,
                    'pre_learning_steps': self.pre_learning_steps,
                    'learning_frequency': self.learning_frequency,
                    'output_loss_freq': self.output_loss_freq
                }
        }

        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(ckpt)
        orbax_checkpointer.save(save_path.resolve(), ckpt, save_args=save_args)
        return

    def update_network(
            self,
            network: TrainState
    ) -> TrainState:
        return network.replace(
            target_params=optax.incremental_update(
                network.params,
                network.target_params,
                self.tau
            )
        )
