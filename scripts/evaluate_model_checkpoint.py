#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic script that dynamically loads the named model from `config/model.json`,
and proceeds to train the banana navigation agent.  Data regarding training
performance and model checkpoints will be output regularly to
`data/<model name>/<experiment id>/` based on the parameters set in
`config/hyperparameters.json`.
"""
import os
import sys

WORK_DIR = os.environ['ROOT_DIR']
sys.path.append(WORK_DIR)

from unityagents import UnityEnvironment
import json
from tqdm import tqdm
from collections import OrderedDict

from src.model_clients.client import ModelClient

UNITY_ENV_PATH = os.environ['UNITY_ENV_PATH']

if __name__ == '__main__':

    env = UnityEnvironment(file_name=UNITY_ENV_PATH)
    brain = env.brains[env.brain_names[0]]

    with open(os.path.join(WORK_DIR, 'config', 'model.json')) as handle:
        model_config = json.load(handle)

    hyperparam_path = os.path.join(
        WORK_DIR, 'data', model_config['model_name'],
        model_config['experiment_id'], 'experiment_info', 'params.json')

    with open(hyperparam_path, 'r') as handle:
        stored_hyperparams = json.load(handle)

    client = ModelClient(nb_actions=brain.vector_action_space_size,
                         nb_state_features=brain.vector_observation_space_size,
                         hyperparams=stored_hyperparams,
                         model_name=model_config['model_name'],
                         experiment_id=model_config['experiment_id'],
                         overwrite_experiment=model_config[
                             'overwrite_experiment'])