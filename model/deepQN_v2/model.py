"""
DQN with fixed targets
"""

import numpy as np
import torch
from model.deepQN_v1.model import Model as ParentModel
from src.base_networks.base_network import Network


class Model(ParentModel):
    def __init__(self, model_name, experiment_id, nb_state_features, nb_actions,
                 train_config):
        super(Model, self).__init__(
            model_name, experiment_id, nb_state_features, nb_actions,
            train_config)

        self.fixed_network = Network(
            nb_features=self.state_size, nb_actions=self.nb_actions,
            params=self.params, seed=self.params['random_seed'])

    def get_sarsa(self):
        experiences = [self.experience_buffer.pop(0) for _ in range(
            0, self.params['batch_size'])]

        states, actions, rewards, next_states = zip(*experiences)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)

        return states, actions, rewards, next_states

    def train_model(self, states, actions, rewards, next_states, next_actions):
        gamma = self.train_config['gamma']

        state_values = self.estimate_action_values(states=next_states,
                                                   fixed=True).detach()
        action_values = state_values.max(dim=1)[0]
        q_targets = rewards + (gamma * action_values)

        self.optimizer.zero_grad()
        expected_state_values = self.estimate_action_values(states=states,
                                                            fixed=False)

        q_expected = expected_state_values[range(0, len(actions)), actions]

        q_delta = self.criterion(q_expected, q_targets.float())
        q_delta.backward()
        self.optimizer.step()

        self.soft_update(src_model=self.network.network,
                         dst_model=self.fixed_network.network,
                         tau=self.params['tau'])

    def estimate_action_values(self, states, fixed):
        states = torch.from_numpy(states).float()
        if fixed:
            action_values_tensor = self.fixed_network.forward(states)
        else:
            action_values_tensor = self.network.forward(states)

        return action_values_tensor

    def soft_update(self, src_model, dst_model, tau):
        for dst_param, src_param in zip(dst_model.parameters(),
                                        src_model.parameters()):
            updated_param = tau*src_param.data + (1.0-tau)*dst_param.data
            dst_param.data.copy_(updated_param)
