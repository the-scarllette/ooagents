from variationalagent import VariationalAgent
from typing import List, Tuple
import numpy as np
import flax
import optax
import jax
import flax.linen as nn
from jax import random, numpy as jnp
from flax.training.train_state import TrainState

class GaussianModule(nn.Module):
    fix_std: bool
    hidden_features: List[int]
    output_features: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for features in self.hidden_features:
            x = nn.Dense(features)(x)
            x = nn.relu(x)

        mu = nn.Dense(self.output_features)(x)
        log_sigma = nn.Dense(self.output_features)(x) if not self.fix_std else jnp.zeros((mu.shape[0], self.output_features))
        return mu, log_sigma
    
class DiscreteModule(nn.Module):
    hidden_features: List[int]
    
    @nn.compact
    def __call__(self, x:jnp.ndarray) -> jnp.ndarray:
        for features in self.hidden_features:
            x = nn.Dense(features)(x)
            x = nn.relu(x)
            
        x = nn.Dense(x)
        return x

class SkillDynamicsModel(nn.Module):
    """
    Skill transition model, Gaussian distribution in continuous spaces and Categorical distribution
    in discrete spaces.
    """
    def __init__(
        self,
        state_feature_dim: int,
        skill_feature_dim: int,
        hidden_features: List[int],
        fix_std: bool,
        xy_prior: bool,
        lr: float=1e-3
    ):
        self.state_feature_dim = state_feature_dim
        self.skill_feature_dim = skill_feature_dim
        self.hidden_features = hidden_features
        self.fix_std = fix_std
        self.xy_prior = xy_prior
        self.lr = lr
        
        self.output_feature_dim = 2 if xy_prior else state_feature_dim
        
        # Create and initialise the model.
        self.model = GaussianModule(
            fix_std=fix_std,
            hidden_features=hidden_features,
            output_features=self.output_feature_dim
        )
        start_key = jax.random.PRNGKey(0)
        self.model_state = TrainState.create(
            apply_fn=self.model.apply,
            params=self.model.init(start_key, jnp.ones((self.state_feature_dim,))),
            tx=optax.adam(learning_rate=self.lr)
        )
        self.model.apply = jax.jit(self.model.apply)
        
    def log_prob(
        self,
        next_state,
        state,
        skill
    ) -> jnp.ndarray:
        x = jnp.concatenate([state, skill], axis=-1)
        mu, log_sigma = self.model.apply(self.model_state.params, x)
        delta = next_state - state
        log_prob = -0.5 * self.state_feature_dim * jnp.log(2 * jnp.pi()) - log_sigma.sum() - 0.5 * ((delta - mu) ** 2 / log_sigma * 2).sum()
        return log_prob
    
class MixtureSkillDynamicsModel:
    pass  

class DadsBatch:
    def __init__(
        self,
        states: np.ndarray,
        next_states: np.ndarray,
        skills: np.ndarray,
    ):
        self.states = states
        self.next_states = next_states
        self.skills = skills
        
    @property
    def states(
        self
    ) -> int:
        return self.states
    
    @property
    def next_states(
        self
    ) -> np.ndarray:
        return self.next_states
    
    @property
    def skills(
        self
    ) -> np.ndarray:
        return self.skills
    
class DadsPolicyBatch(DadsBatch):
    def __init__(
        self,
        states: np.ndarray,
        next_states: np.ndarray,
        skills: np.ndarray,
        rewards: np.ndarray
    ):
        super().__init__(
            states=states,
            next_states=next_states,
            skills=skills
        )
        self.rewards = rewards
        
    @property
    def rewards(self) -> np.ndarray:
        return self.rewards

class DadsAgent(VariationalAgent):

    def __init__(
        self,
        actions: int | List[int] | Tuple[int],
        state_feature_dim: int,
        skill_feature_dim: int,
        hidden_features: List[int],
        discrete: bool,
        fix_std: bool,
        xy_prior: bool  
    ):
        super().__init__(actions)
        self.state_feature_dim = state_feature_dim
        self.skill_feature_dim = skill_feature_dim
        self.hidden_features = hidden_features
        self.discrete = discrete
        self.fix_std = fix_std
        self.xy_prior = xy_prior
        
        self.dynamics_model = SkillDynamicsModel(
            state_feature_dim=state_feature_dim,
            skill_feature_dim=skill_feature_dim,
            hidden_features=hidden_features,
            discrete=discrete,
            fix_std=fix_std,
            xy_prior=xy_prior
        )
    
    def generate_mb_indices(self, total):
        """
        Method for constructing minibatches for SGD. Returns randomised indices for minibatches.
        """
        minibatch_start = np.arange(0, total, self.mb_size)
        indices = np.arange(total, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[start:start + self.mb_size] for start in minibatch_start]
        return batches
    
    def learn_representation(
        self,
        batch: DadsBatch,
        mb_size: int,
        epochs: int,
    ) -> None:
        
        def _train_network(states: np.ndarray, next_states: np.ndarray, skills: np.ndarray) -> Tuple[jnp.ndarray]:
            log_probs = self.dynamics_model.log_prob(next_states, states, skills)
            loss = - log_probs.mean()
            
        
        assert batch.states.shape[0] % mb_size == 0
        states = batch.states
        next_states = batch.next_states
        skills = batch.skills
        
        for epoch in range(epochs):
            # Generate minibatches
            mb_indices = self.generate_mb_indices(states.shape[0])
            
            for mb in mb_indices:
                mb_states = states[mb, ...]
                mb_next_states = next_states[mb, ...]
                mb_skills = skills[mb, ...]
                
                
                
    def compute_reward(
        self,
        batch: DadsBatch,
    ) -> DadsPolicyBatch:
        rewards = - self.dynamics_model.log_prob(batch.next_states, batch.states, batch.skills)
        return DadsPolicyBatch(batch.states, batch.next_states, batch.skills, rewards)
    