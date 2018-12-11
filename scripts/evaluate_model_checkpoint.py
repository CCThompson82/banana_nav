#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic script that dynamically loads the named model from `config/model.json`,
and proceeds to evaluate chekpoints made during the training of the model.
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

    with open(os.path.join(WORK_DIR, 'config', 'eval.json')) as handle:
        eval_config = json.load(handle)

    hyperparam_path = os.path.join(
        WORK_DIR, 'data', eval_config['model_name'],
        eval_config['experiment_id'], 'experiment_info', 'params.json')

    with open(hyperparam_path, 'r') as handle:
        stored_hyperparams = json.load(handle)

    client = ModelClient(nb_actions=brain.vector_action_space_size,
                         nb_state_features=brain.vector_observation_space_size,
                         hyperparams=stored_hyperparams,
                         model_name=eval_config['model_name'],
                         experiment_id=eval_config['experiment_id'],
                         overwrite_experiment='EVAL_MODE')

    checkpoint_dir = os.path.join(
        WORK_DIR, 'data', eval_config['model_name'],
        eval_config['experiment_id'], 'checkpoints')
    checkpoint_set = os.listdir(checkpoint_dir)
    checkpoint_set = ['ckpt_0.pth'] + checkpoint_set

    pbar = tqdm(total=len(checkpoint_set)*int(eval_config['nb_evaluations']))
    for checkpoint in checkpoint_set:

        if checkpoint != 'ckpt_0.pth':
            client.restore_checkpoint(checkpoint)

        trial = eval_config['evaluation_id']
        for episode in range(eval_config['nb_evaluations']):
            pbar.set_postfix(
                ordered_dict=OrderedDict(
                    [('checkpoint', checkpoint.split('.')[0]),
                     ('trial episode', episode),
                     ('mean episode score', client.mean_eval_score(
                         checkpoint, str(trial)))]))
            pbar.update()

            env_info = env.reset(train_mode=True)[brain.brain_name]
            state = env_info.vector_observations[0]

            while not (env_info.local_done[0] or env_info.max_reached[0]):
                action = client.get_next_action(state=state)
                env_info = env.step(action)[brain.brain_name]
                reward = env_info.rewards[0]
                next_state = env_info.vector_observations[0]

                client.store_reward(reward)
                state = next_state
            client.record_eval_episode_score(str(trial), checkpoint)











