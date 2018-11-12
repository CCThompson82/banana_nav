#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic script that dynamically loads the named model from `config/model.json`,
and proceeds to train the banana navigation agent.  Data regarding training
performance and model checkpoints will be output regularly to
`data/<model name>/<experiment id>/` based on the parameters set in
`config/training.json`.
"""
import os
import sys
WORK_DIR = os.environ['ROOT_DIR']
sys.path.append(WORK_DIR)

import json
import numpy as np

from src.base_models.base_model import ModelClient

UNITY_ENV_PATH = os.environ['UNITY_ENV_PATH']

if __name__ == '__main__':

    with open(os.path.join(WORK_DIR, 'config', 'model.json')) as handle:
        model_config = json.load(handle)
    client = ModelClient(**model_config)

    print(client.get_next_action(np.zeros(37)))
