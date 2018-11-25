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
from tqdm import tqdm
from unityagents import UnityEnvironment

from src.model_clients.client import ModelClient

UNITY_ENV_PATH = os.environ['UNITY_ENV_PATH']

if __name__ == '__main__':

    env = UnityEnvironment(file_name=UNITY_ENV_PATH)
    brain = env.brains[env.brain_names[0]]

    with open(os.path.join(WORK_DIR, 'config', 'model.json')) as handle:
        model_config = json.load(handle)
    with open(os.path.join(WORK_DIR, 'config', 'train.json')) as handle:
        train_config = json.load(handle)

    client = ModelClient(nb_actions=brain.vector_action_space_size,
                         nb_state_features= brain.vector_action_space_size,
                         train_config=train_config,
                         **model_config)

    # build buffer with by running episodes
    pbar = tqdm(total=train_config['max_episodes'])
    while not client.training_finished(train_config):
        pbar.update()
        env_info = env.reset(train_mode=True)[brain.brain_name]
        state = env_info.vector_observations

        while not (env_info.local_done[0] or env_info.max_reached[0]):

            action = client.get_next_action(state=state)
            env_info = env.step(action)[brain.brain_name]
            reward = env_info.rewards[0]
            next_state = env_info.vector_observations[0]

            client.store_experience(
                experience=(state, action, reward, next_state))
            client.store_reward(reward)
            state = next_state

            # if buffer is acceptable, train model
            if not client.check_training_status(
                    min_buffer_size=train_config['min_buffer_size']):
                continue
            client.train_model(gamma=train_config['gamma'])

        client.record_episode_score()

        if client.episode_count % train_config['checkpoint_frequency'] == 0:
            client.checkpoint_model()

