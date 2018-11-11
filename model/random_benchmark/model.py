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
    def __init__(self, model_name, experiment_id):
        with open(os.path.join(WORK_DIR, 'model', model_name, experiment_id,
                               "params.json")) as handle:
            self.params = json.load(handle)
            self.nb_actions = 4
            self.state_size = 37

    def next_action(self, state):
        return np.random.randint(0, self.nb_actions)
