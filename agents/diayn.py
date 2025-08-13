import flax
import flax.linen as nn
from flax.training.train_state import TrainState
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from pathlib import Path
import random
from stable_baselines3.common.buffers import ReplayBuffer
from typing import List, Tuple

from agents.sac import SAC
from agents.variationalagent import VariationalAgent

class DiscriminatorNetwork(nn.Module):
    action_dim: int
    shape: List[int]

    @nn.compact
    def __call__(
            self,
            x: jnp.ndarray
    ) -> jnp.ndarray:
        for size in self.shape:
            x = nn.Dense(size)(x)
            x = nn.relu(x)

        x = nn.Dense(self.action_dim)(x)
        x = nn.softmax(x)
        return x

class PolicyNetwork(nn.Module):
    action_dim: int
    shape: List[int]

    @nn.compact
    def __call__(
            self,
            x: jnp.ndarray
    ) -> jnp.ndarray:
        for size in self.shape:
            x = nn.Dense(size)(x)
            x = nn.relu(x)

        x = nn.Dense(self.action_dim)(x)
        return x

class TrainState(TrainState):
    target_params: flax.core.FrozenDict

class DIAYN(VariationalAgent):

    def __init__(
            self,
            environment: gym.Env,
            continuous_actions: bool,
            actions: int | List[int] | Tuple[int],
            policy_shape: None|List[int]=None,
            discriminator_shape: None|List[int]=None,
            buffer_size: int=1_000_000,
            reward_scale: float=1.0,
            learning_rate: float=3e-4,
            gamma: float=0.99,
            tau: float=1.0,
            minibatch_size: int = 256,
            skill_selection_frequency: int=10,
            pre_learning_steps: int=5e3,
            learning_frequency: int=500,
            target_network_update_freq: int = 500,
            discrete_skills: bool = True,
            num_skills: None | int = 3,
            output_loss_frequency: int=1e4
    ):
        super().__init__(actions)

        self.policy_shape: None|List[int] = policy_shape
        if self.policy_shape is None:
            self.policy_shape = [256, 256]
        self.discriminator_shape: None|List[int] = discriminator_shape
        if self.discriminator_shape is None:
            self.discriminator_shape = [256, 256]

        self.buffer_size: int = buffer_size
        self.minibatch_size: int = minibatch_size
        self.learning_rate: float = learning_rate
        self.discrete_skills: bool = discrete_skills
        self.num_skills: int = num_skills
        if not self.discrete_skills:
            self.num_skills = 1

        self.continuous_actions: bool = continuous_actions

        self.tau: float = tau

        self.skill_selection_frequency: int = skill_selection_frequency
        self.pre_learning_steps: int = pre_learning_steps
        self.learning_frequency: int = learning_frequency
        self.output_loss_frequency: int = output_loss_frequency
        self.target_network_update_freq: int = target_network_update_freq

        self.state_shape: Tuple[int, ...] = environment.observation_space.shape
        self.state_skill_shape: Tuple[int] = (self.state_shape[0] + self.num_skills,)


        self.policy_network: PolicyNetwork = PolicyNetwork(
            action_dim=self.num_skills,
            shape=self.policy_shape,
        )
        self.discriminator_network: DiscriminatorNetwork = DiscriminatorNetwork(
            action_dim=self.num_skills,
            shape=self.discriminator_shape,
        )

        start_key = jax.random.PRNGKey(0)
        obs, _ = environment.reset()
        self.policy: TrainState = TrainState.create(
            apply_fn=self.policy_network.apply,
            params=self.policy_network.init(start_key, obs),
            target_params=self.policy_network.init(start_key, obs),
            tx=optax.adam(learning_rate=self.learning_rate)
        )

        sample_skill = jnp.zeros(self.num_skills)
        self.policy.apply = jax.jit(self.policy_network.apply)
        self.discriminator: TrainState = TrainState.create(
            apply_fn=self.discriminator_network.apply,
            params=self.discriminator_network.init(start_key, obs),
            target_params=self.discriminator_network.init(start_key, obs),
            tx=optax.adam(learning_rate=self.learning_rate)
        )
        self.discriminator.apply = jax.jit(self.discriminator_network.apply)

        self.discriminator_replay_buffer: ReplayBuffer = ReplayBuffer(
            self.buffer_size,
            environment.observation_space,
            gym.spaces.Space(sample_skill.shape, float),
            "cpu",
            handle_timeout_termination=False
        )

        self.policy_replay_buffer: ReplayBuffer = ReplayBuffer(
            self.buffer_size,
            environment.observation_space,
            gym.spaces.Space(sample_skill.shape, float),
            "cpu",
            handle_timeout_termination=False
        )

        state_skill_sample = jnp.concatenate([[obs], [sample_skill]], axis=-1)
        self.action_policy: SAC = SAC(
            environment,
            continuous_actions,
            self.policy_shape,
            buffer_size,
            reward_scale,
            self.learning_rate,
            gamma,
            self.tau,
            self.minibatch_size,
            self.pre_learning_steps,
            self.learning_frequency,
            self.output_loss_frequency,
            self.target_network_update_freq,
            state_shape=state_skill_sample.shape,
            observation_sample=state_skill_sample
        )

        self.training_steps: int = 0
        return

    def choose_action(self, state: np.ndarray, possible_actions: List[int]|None=None,
                      no_random: bool=False) -> int|float:
        if no_random:
            return self.actions[0]
        return random.choice(self.actions)

    def learn(self, state: np.ndarray, action: int|float, reward: float, next_state: np.ndarray,
              terminal: bool=False, next_state_possible_actions: List[int]|None=None) -> None:
        return None

    def learn_representation(
            self,
            state: np.ndarray,
            action: int|float,
            reward: float,
            next_state: np.ndarray,
            terminal: bool=False,
            next_state_possible_actions: List[int]|None=None
    ) -> None:
        return

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

    @staticmethod
    def load(environment: gym.Env, load_path: Path) -> 'Agent':
        with load_path.open('r') as f:
            agent_data = json.load(f)
        return Agent(agent_data['actions'])

    def sample_skill(
            self
    ) -> jnp.ndarray:
        skill = jnp.zeros(self.num_skills)
        if self.discrete_skills:
            skill_index = random.randint(0, self.num_skills - 1)
            skill[skill_index] = 1
            return skill

        skill[0] = random.uniform(0, 1)
        return skill

    def save(self, save_path: Path) -> None:
        with save_path.open('w') as f:
            json.dump({'actions': self.actions}, f)
        return

    def train_skills(
            self,
            environment: gym.Env,
            training_steps: int
    ):
        def train_discriminator(
                states: np.ndarray,
                skills: np.ndarray
        ) -> Tuple[jnp.ndarray, TrainState]:
            def cross_entropy_loss(params) -> Tuple[jnp.ndarray, jnp.ndarray]:
                skills_outputted = self.discriminator_network.apply(params, states)
                log_skills = jnp.log(skills_outputted)
                return optax.losses.safe_softmax_cross_entropy(log_skills, skills), skills_outputted

            def mse_loss(params) -> Tuple[jnp.ndarray, jnp.ndarray]:
                skills_outputted = self.discriminator_network.apply(params, states)
                return ((skills_outputted - skills)**2).mean(), skills_outputted

            loss_func = cross_entropy_loss
            if not self.discrete_skills:
                loss_func = mse_loss

            (loss, _), grads = jax.value_and_grad(loss_func, has_aux=True)(self.discriminator.params)
            new_discriminator_state = self.discriminator.apply_gradients(grads=grads)
            return loss, new_discriminator_state
        
        done: bool = True
        current_steps: int = 0

        log_skill_prob: float = 0.0
        if self.discrete_skills:
            log_skill_prob = np.log(1/self.num_skills)

        while current_steps < training_steps:
            if done:
                state, _ = environment.reset()
                training_skill = self.sample_skill()
                training_skill_index = np.where(training_skill == 1)[0]

            action = self.action_policy.choose_action(jnp.concatenate([[state], [training_skill]], axis=-1))
            next_state, _, done, truncated, _ = environment.step(action)

            p_z_array = self.discriminator_network.apply(next_state)
            approx_skill_prob = p_z_array[training_skill_index]

            skill_reward = jnp.log(approx_skill_prob) - log_skill_prob

            self.action_policy.learn(state, action, skill_reward[0], next_state, done)

            # add transition to replay buffer
            self.discriminator_replay_buffer.add(
                state,
                next_state,
                np.array(training_skill),
                np.array([0]),
                np.array([done]),
                None
            )

            # update discriminator with SGD

            current_steps += 1

            if (current_steps > self.pre_learning_steps) and (current_steps % self.learning_frequency == 0):
                transitions = self.discriminator_replay_buffer.sample(self.minibatch_size)
                discriminator_loss, self.discriminator = train_discriminator(
                    transitions.next_observations.numpy(), # Next states
                    transitions.actions.numpy() # Skills
                )

            if current_steps % self.target_network_update_freq == 0:
                self.discriminator = self.discriminator.replace(
                    target_params=optax.incremental_update(
                        self.discriminator.params,
                        self.discriminator.target_params,
                        self.tau
                    )
                )

        return
