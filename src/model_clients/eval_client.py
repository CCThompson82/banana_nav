"""
Model object that abstracts dynamic model versioning and directory management.
"""

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

    @property
    def mean_episode_score(self):
        pass

    @property
    def best_episode_score(self):
        pass
