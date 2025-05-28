import json
import numpy as np
from pathlib import Path
import random

from agents.agent import Agent
from typing import Dict, List, Tuple, Type

class QLearningAgent(Agent):

    def __init__(
            self,
            actions: int|List[int],
            alpha: float,
            epsilon: float,
            gamma: float
    ):
        super().__init__(actions)

        self.num_actions = len(self.actions)

        self.alpha: float = alpha
        self.epsilon: float = epsilon
        self.gamma: float = gamma

        self.q_values: Dict[str, np.ndarray] = {}
        return

    def choose_action(
            self,
            state: np.ndarray,
            possible_actions: np.ndarray|None=None,
            no_random: bool=False
    ):
        if possible_actions is None:
            possible_actions = self.actions

        if no_random or (random.uniform(0, 1) <= self.epsilon):
            return random.choice(possible_actions)

        q_values = self.get_q_values(state)
        max_value = -np.inf
        chosen_actions = []
        for action in possible_actions:
            q_value = q_values[action]
            if q_value > max_value:
                max_value = q_value
                chosen_actions  = [action]
            elif q_value == max_value:
                chosen_actions.append(action)

        return random.choice(chosen_actions)

    def get_q_values(
            self,
            state: np.ndarray,
    ) -> np.ndarray:
        state_str = self.state_to_str(state)

        try:
            q_values = self.q_values[state_str]
        except KeyError:
            self.q_values[state_str] = np.zeros(self.num_actions, dtype=np.float32)
            q_values = self.q_values[state_str]

        return q_values

    def learn(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            terminal: bool=False,
            next_possible_actions: np.ndarray|None=None
    ):
        if next_possible_actions is None:
            next_possible_actions = self.actions
        state_str = self.state_to_str(state)

        q_value = self.get_q_values(state)[action]
        next_state_q_values = self.get_q_values(next_state)[next_possible_actions]

        max_next_value = 0.0
        if not terminal:
            max_next_value = np.max(next_state_q_values)

        self.q_values[state_str][action] += self.alpha * (reward + (self.gamma * max_next_value) - q_value)
        return

    @staticmethod
    def load(
            load_path: Path
    ) -> 'QLearningAgent':
        with open(load_path, 'r') as f:
            agent_data = json.load(f)

        q_learning_agent = QLearningAgent(
            agent_data['actions'],
            agent_data['alpha'],
            agent_data['epsilon'],
            agent_data['gamma']
        )

        q_learning_agent.q_values = {
            state: np.array(agent_data['q_values'][state]) for state in agent_data['q_values']
        }

        return q_learning_agent

    def save(
            self,
            save_path: Path
    ):
        agent_data = {
            'actions': self.actions,
            'alpha': self.alpha,
            'epsilon': self.epsilon,
            'gamma': self.gamma
            'q_values': {
                state: self.q_values[state].tolist() for state in self.q_values
            }
        }

        with open(save_path, 'w') as f:
            json.dump(agent_data, f)
        return

    @staticmethod
    def state_to_str(
            state: np.ndarray
    ) -> str:
        return np.array2string(state)
