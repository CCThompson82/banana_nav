"""
DQN_v1: 3 layer neural network q value approximation with experience replay
"""

import numpy as np
from model.deepQN_v0.model import Model as ParentModel
import torch


class Model(ParentModel):
    def __init__(self, model_name, experiment_id, nb_state_features, nb_actions,
                 hyperparams, overwrite_experiment):
        super(Model, self).__init__(
            model_name=model_name,
            experiment_id=experiment_id,
            nb_state_features=nb_state_features,
            nb_actions=nb_actions,
            hyperparams=hyperparams,
            overwrite_experiment=overwrite_experiment)

    def get_sarsa(self):

        experience_index = np.random.choice(
            range(len(self.experience_buffer)), self.params['batch_size'],
            replace=False)

        experiences = [self.experience_buffer.pop(index) for index in sorted(
            experience_index, reverse=True)]

        states, actions, rewards, next_states = zip(*experiences)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)

        return states, actions, rewards, next_states

    def check_training_status(self):
        status = (len(self.experience_buffer) >=
                  self.hyperparams['min_buffer_size'])
        return status

    def train_model(self, states, actions, rewards, next_states, next_actions):
        gamma = self.hyperparams['gamma']

        state_values = self.estimate_action_values(states=next_states).detach()
        action_values = state_values.max(dim=1)[0]
        q_targets = rewards + (gamma * action_values)

        self.optimizer.zero_grad()
        expected_state_values = self.estimate_action_values(states=states)

        q_expected = expected_state_values[range(0, len(actions)), actions]

        q_delta = self.criterion(q_expected, q_targets.float())
        q_delta.backward()
        self.optimizer.step()

    def estimate_action_values(self, states):
        states = torch.from_numpy(states).float()
        action_values_tensor = self.network.forward(states)

        return action_values_tensor
