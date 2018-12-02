"""
Vanilla DQN
"""
import os
import sys


WORK_DIR = os.environ['ROOT_DIR']
sys.path.append(WORK_DIR)

import json
import numpy as np
from src.base_model.base_model import BaseModel
from src.base_networks.base_network import Network
import torch
nn = torch.nn


class Model(BaseModel):
    def __init__(self, model_name, experiment_id, nb_state_features, nb_actions,
                 train_config):
        super(Model, self).__init__(model_name=model_name,
                                    experiment_id=experiment_id,
                                    train_config=train_config)
        with open(os.path.join(WORK_DIR, 'model', model_name, experiment_id,
                               "params.json")) as handle:
            self.params = json.load(handle)
        self.nb_actions = nb_actions
        self.state_size = nb_state_features

        self.experience_buffer = []

        self.network = Network(
            nb_features=self.state_size, nb_actions=self.nb_actions,
            params=self.params, seed=self.params['random_seed'])
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            params=self.network.parameters(),
            lr=self.params['init_learning_rate'])

    def next_action(self, state, epsilon):
        state_tensor = torch.from_numpy(state).float()

        action_values = self.network.network.forward(state_tensor).data.numpy()
        action = np.argmax(action_values, axis=-1)

        if np.random.rand() < epsilon:
            action = np.random.randint(0, self.nb_actions)

        return action

    def next_max_action(self, state):
        state_tensor = torch.from_numpy(state).float()

        action_values = self.network.network.forward(
            state_tensor).detach().data.numpy()
        action = np.argmax(action_values, axis=-1)

        return action

    def store_experience(self, experience):
        self.experience_buffer.append(experience)

    def get_sarsa(self):
        return self.experience_buffer.pop(0)

    def train_model(self, state, action, reward, next_state, next_action):
        gamma = self.train_config['gamma']
        q_expected = self.estimate_q(state, action).detach()

        self.optimizer.zero_grad()
        q_hat = reward + (gamma * self.estimate_q(state=next_state,
                                                  action=next_action))
        q_delta = self.criterion(q_hat, q_expected)
        q_delta.backward()
        self.optimizer.step()

    def estimate_q(self, state, action, **kwargs):
        state = torch.from_numpy(state).float()
        action_values_tensor = self.network.forward(state)
        return action_values_tensor[action]

    def update_model_weights(self, loss):
        pass

    def get_epsilon(self, step_count):
        if step_count == 0:
            epsilon = 1.0
        else:
            epsilon = (1.0/step_count)**(1/self.params['epsilon_root_factor'])
        return np.round(epsilon, 3)

    def terminate_training_status(self, episode_count, **kwargs):
        return episode_count >= self.train_config['max_episodes']

    def checkpoint_model(self, episode_count):
        checkpoint_fn = os.path.join(self.checkpoint_dir, '{}.json'.format(
            episode_count))
        weights = {}
        with open(checkpoint_fn, 'w') as out:
            json.dump(weights, out)

    def check_training_status(self):
        status = (len(self.experience_buffer) >=
                  self.train_config['min_buffer_size'])
        return status


