"""
DQN with experience replay and fixed target approximation
"""

import numpy as np
from model.deepQN_v2.model import Model as ParentModel


class Model(ParentModel):
    def __init__(self, model_name, experiment_id, nb_state_features, nb_actions,
                 hyperparams):
        super(Model, self).__init__(
            model_name, experiment_id, nb_state_features, nb_actions,
            hyperparams)

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
