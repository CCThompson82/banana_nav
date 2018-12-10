"""
Model object that abstracts dynamic model versioning and directory management.
"""
import os
from pydoc import locate
import numpy as np


class ModelClient(object):
    def __init__(self, nb_actions, nb_state_features, model_name,
                 experiment_id, overwrite_experiment, hyperparams):
        self.hyperparams = hyperparams
        self.nb_actions = nb_actions
        self.state_shape = nb_state_features

        self.model = self.load_model(model_name=model_name,
                                     experiment_id=experiment_id,
                                     overwrite_experiment=overwrite_experiment,
                                     nb_actions=nb_actions,
                                     nb_state_features=nb_state_features,
                                     hyperparams=hyperparams)
        self.step_count = 0
        self.episode_count = 0
        self.episode_score = 0

    def load_model(self, model_name, experiment_id, overwrite_experiment,
                   nb_actions, nb_state_features, hyperparams):
        Model = locate('model.{}.model.Model'.format(model_name))
        model = Model(model_name=model_name,
                      experiment_id=experiment_id,
                      overwrite_experiment=overwrite_experiment,
                      nb_state_features=nb_state_features,
                      nb_actions=nb_actions,
                      hyperparams=hyperparams)
        return model

    def get_next_action(self, state):
        self.step_count += 1
        return self.model.next_action(state=state, epsilon=self.epsilon)

    def get_next_max_action(self, state):
        return self.model.next_max_action(state)

    def store_experience(self, experience):
        self.model.store_experience(experience)

    def get_sarsa(self):
        return self.model.get_sarsa()

    def check_training_status(self):
        status = self.model.check_training_status()
        return status

    def train_model(self):

        states, actions, rewards, next_states = self.get_sarsa()
        next_actions = self.get_next_max_action(next_states)
        self.model.train_model(
            states, actions, rewards, next_states, next_actions)

    @property
    def epsilon(self):
        return self.model.get_epsilon(step_count=self.step_count)

    def training_finished(self):
        return self.model.terminate_training_status(
            step_count=self.step_count,
            episode_count=self.episode_count)

    def store_reward(self, reward):
        self.episode_score += reward

    def record_episode_score(self):
        try:
            arr = np.load(self.model.results_filename)
            arr = np.concatenate([arr, np.array([self.episode_score])])
        except FileNotFoundError:
            arr = np.array([self.episode_score])
        np.save(self.model.results_filename, arr)
        self.reset_episode()

    def reset_episode(self):
        self.episode_score = 0
        self.episode_count += 1

    def checkpoint_model(self):
        self.model.checkpoint_model(episode_count=self.episode_count)

    @property
    def mean_episode_score(self):
        try:
            arr = np.load(self.model.results_filename)
        except FileNotFoundError:
            return 0

        if len(arr) < 100:
            return np.round(np.mean(arr), 3)

        return np.round(np.mean(arr[-100:]), 3)

    @property
    def best_episode_score(self):
        try:
            arr = np.load(self.model.results_filename)
        except FileNotFoundError:
            return 0
        return np.round(np.max(arr), 3)

    @property
    def buffer_size(self):
        try:
            return len(self.model.experience_buffer)
        except AttributeError:
            return 0.0

    def checkpoint_step(self, frequency):
        return self.episode_count % frequency == 0

    def mean_eval_score(self, checkpoint, trial):

        checkpoint = '{}.npy'.format(checkpoint.split('.')[0])
        try:
            arr = np.load(
                os.path.join(self.model.evaluation_dir, trial, checkpoint))
            return np.mean(arr)
        except FileNotFoundError:
            return 0.0

    def record_eval_episode_score(self, trial, checkpoint):
        checkpoint_fn = '{}.npy'.format(checkpoint.split('.')[0])
        eval_fn = os.path.join(self.model.evaluation_dir, trial, checkpoint_fn)
        try:
            arr = np.load(eval_fn)
            arr = np.concatenate([arr, np.array([self.episode_score])])
        except FileNotFoundError:
            arr = np.array([self.episode_score])
        np.save(eval_fn, arr)
        self.reset_episode()

    def restore_checkpoint(self, checkpoint):
        self.model.restore_checkpoint(checkpoint=checkpoint)



