from pathlib import Path

from agents import Agent
from utils import print_progress_bar

def train_agent(environment, agent: Agent,
                timesteps: int,
                evaluate_frequency: int=0,
                agent_save_path: None|Path=None,
                all_actions_valid: bool=True,
                progress_bar: bool=False):
    done = True
    epoch_returns = []
    training_returns = []

    for steps in range(timesteps):
        if progress_bar:
            print_progress_bar(steps, timesteps,
                               prefix='Agent Training: ', suffix='Complete')


    return
