from typing import List
from typing import List, Tuple
import flax.linen as nn
import jax.numpy as jnp


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
    
