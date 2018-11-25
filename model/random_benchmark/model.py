"""
Random action agent model
"""
import os
import sys
WORK_DIR = os.environ['ROOT_DIR']
sys.path.append(WORK_DIR)

import json
import numpy as np


class Model(object):
    def __init__(self, model_name, experiment_id, brain):
        with open(os.path.join(WORK_DIR, 'model', model_name, experiment_id,
                               "params.json")) as handle:
            self.params = json.load(handle)
        self.nb_actions = brain.vector_action_space_size
        self.state_size = brain.vector_observation_space_size

        self.experience_buffer = []

    def next_action(self, state, epsilon):
        return np.random.randint(0, self.nb_actions)

    def next_max_action(self, state):
        return np.random.randint(0, self.nb_actions)

    def store_experience(self, experience):
        self.experience_buffer.append(experience)

    def get_experience_from_buffer(self):
        return self.experience_buffer.pop(0)

    def estimate_q(self, state, action, fixed):
        return np.random.rand()

    def update_model_weights(self, loss):
        pass

    def get_epsilon(self, step_count):
        return 1.0/step_count

    def terminate_training_status(self, train_config, step_count):
        return step_count >= train_config['max_training_steps']



