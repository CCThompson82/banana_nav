"""
Random action agent model
"""
import os
import sys
WORK_DIR = os.environ['ROOT_DIR']
sys.path.append(WORK_DIR)

import json


class Model(object):
    def __init__(self, model_name, experiment_id):
        with open(os.path.join(WORK_DIR, 'model', model_name, experiment_id,
                               "params.json")) as handle:
            self.params = json.load(handle)
