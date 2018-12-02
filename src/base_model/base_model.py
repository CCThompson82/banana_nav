

import os
import sys
import json
import shutil

ROOT_DIR = os.environ['ROOT_DIR']
sys.path.append(ROOT_DIR)


class BaseModel(object):
    def __init__(self, model_name, experiment_id, overwrite_model, hyperparams):
        self.model_name = model_name
        self.experiment_id = experiment_id

        self.model_dir = os.path.join(
            ROOT_DIR, 'data', model_name, experiment_id)
        self.results_dir = os.path.join(self.model_dir, 'results')
        self.results_filename = os.path.join(self.results_dir,
                                             'episode_scores.npy')
        self.checkpoint_dir = os.path.join(self.model_dir, 'checkpoints')
        self.experiment_info_dir = os.path.join(
            self.model_dir, 'experiment_info')

        if not os.path.isdir(self.model_dir):
            self.create_directory_structure()

        elif overwrite_model:
            shutil.rmtree(self.model_dir)
            self.create_directory_structure()

        else:
            raise IOError(
                'An experiment for {}: {} already exists.  Set overwrite to '
                'True in  `config/hyperparameters.json` if you wish to overwrite the '
                'previous experiment.'.format(self.model_name,
                                              self.experiment_id))

        self.dump_experiment_info(hyperparams)
        self.hyperparams = hyperparams

    def create_directory_structure(self):
        os.makedirs(self.model_dir)
        os.mkdir(os.path.join(self.model_dir, 'results'))
        os.mkdir(os.path.join(self.model_dir, 'checkpoints'))
        os.mkdir(os.path.join(self.model_dir, 'experiment_info'))

    def dump_experiment_info(self, hyperparams):
        hyperparams['model_name'] = self.model_name
        hyperparams['experiment_id'] = self.experiment_id

        filename = os.path.join(self.experiment_info_dir, 'params.json')
        with open(filename, 'w') as out:
            json.dump(hyperparams, out)
