

import os
import sys

ROOT_DIR = os.environ['ROOT_DIR']
sys.path.append(ROOT_DIR)


class BaseModel(object):
    def __init__(self, model_name, experiment_id):
        self.model_name = model_name
        self.experiment_id = experiment_id

        self.model_dir = os.path.join(
            ROOT_DIR, 'data', model_name, experiment_id)
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
            os.mkdir(os.path.join(self.model_dir, 'results'))
            os.mkdir(os.path.join(self.model_dir, 'checkpoints'))
