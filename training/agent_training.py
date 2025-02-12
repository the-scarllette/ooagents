from pathlib import Path
import random

from agents.agent import Agent
from utils.load_agent import load_agent
from utils.progress_bar import print_progress_bar

def run_epoch(environment, agent: Agent,
              epoch: int=50,
              seed: None|int=None,
              all_actions_valid: bool=True,
              progress_bar: bool=False) -> float:
    done = True
    possible_actions = None
    epoch_return = 0.0
    if seed is None:
        seed = random.randint(1, 10_000)

    for steps in range(epoch):
        if progress_bar:
            print_progress_bar(steps, epoch,
                               prefix="     Running Epoch: ", suffix="Complete")

        if done:
            state, _ = environment.reset(seed=seed)
            seed += 1
            if not all_actions_valid:
                possible_actions = environment.get_possible_actions()

        action = agent.choose_action(state, possible_actions, True)
        next_state, reward, done, _ = environment.step(action)
        epoch_return += reward

        if all_actions_valid:
            agent.learn(state, action, reward, next_state, done)
        else:
            possible_actions = environment.get_possible_actions()
            agent.learn(state, action, reward, next_state, done,
                        possible_actions)

        state = next_state

    return epoch_return

def train_agent(environment, agent: Agent,
                timesteps: int,
                evaluate_frequency: int=0, epoch: int=50,
                agent_save_path: None|Path=None,
                all_actions_valid: bool=True,
                seed_evaluations: bool=False,
                progress_bar: bool=False) -> Agent:
    done = True
    possible_actions = None
    epoch_returns = []
    training_returns = []

    for steps in range(timesteps):
        if progress_bar:
            print_progress_bar(steps, timesteps,
                               prefix='Agent Training: ', suffix='Complete')

        if (evaluate_frequency > 0) and (steps % evaluate_frequency == 0):
            if agent_save_path is None:
                raise ValueError("Must provide a path to save and load agent for evaluation")
            agent.save(agent_save_path)
            evaluate_agent = load_agent(agent_save_path, type(agent))

            seed = steps if seed_evaluations else None

            epoch_return = run_epoch(environment, evaluate_agent,
                                     epoch, seed, all_actions_valid, progress_bar)

            epoch_returns.append(epoch_return)

        if done:
            state, _ = environment.reset()
            if not all_actions_valid:
                possible_actions = environment.get_possible_actions()

        action = agent.choose_action(state, possible_actions)
        next_state, reward, done, _ = environment.step(action)

        if all_actions_valid:
            agent.learn(state, action, reward, next_state, done)
        else:
            possible_actions = environment.get_possible_actions()
            agent.learn(state, action, reward, next_state, done,
                        possible_actions)

        state = next_state
        training_returns.append(reward)

    if agent_save_path is not None:
        agent.save(agent_save_path)

    return agent
