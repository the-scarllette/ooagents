# Object Orientated Agents
A bank of reinforcement learning agents implemented in object orientated.
Every agent follows the same structure so agents can be 'plugged-into' environments.

## Usage

Every agent inherits from the `Agent` superclass.
Every agent has a `choose_action` method and a `learn` method with the following signatures:

```
def choose_action(self,
                  state: np.ndarray,
                  possible_actions: List[int]|None=None,
                  no_random: bool=False) -> int|float
                  
def learn(self,
          state: np.ndarray,
          action: int|float,
          next_state: np.ndarray,
          reward: float,
          terminal: bool=False,
          next_state_possible_actions: List[int]|None=None) -> None
```

So the agent-environment loop (for any agent and environment) can be written as:

```
done = False
state, _ = environment.reset()

while not done:
    action = agent.choose_action(state)
    next_state, rewad, done, _, _ = environment.step(action)
    agent.learn(state, action, reward, next_state, done)
    state = next_state
```

## Current Agents

Agents are implemented using [JAX](https://docs.jax.dev/en/latest/index.html).

### Q-Learning
The Q-Learning agent from [Watkins C.](https://www.cs.rhul.ac.uk/~chrisw/thesis.html).
This agent is a tabular agent suited to small, discrete environments.

### DQN
The DQN agent from [Mnih, V., Kavukcuoglu, K., Silver, D. et al.](https://www.nature.com/articles/nature14236#citeas)
is implemented.

## License

[MIT](https://choosealicense.com/licenses/mit/)