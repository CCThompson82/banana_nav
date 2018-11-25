"""
Model object that abstracts dynamic model versioning and directory management.
"""

from pydoc import locate


class ModelClient(object):
    def __init__(self, brain, model_name, experiment_id, train_config):
        self.train_config=train_config
        self.nb_actions = brain.vector_action_space_size
        self.state_shape = brain.vector_observation_space_size

        self.model = self.load_model(model_name, experiment_id, brain,
                                     train_config)
        self.step_count = 0
        self.episode_count = 0
        self.episode_score = 0

    def load_model(self, model_name, experiment_id, brain, train_config):
        Model = locate('model.{}.model.Model'.format(model_name))
        model = Model(model_name, experiment_id, brain, train_config)
        return model

    def get_next_action(self, state):
        self.step_count += 1
        return self.model.next_action(state=state, epsilon=self.epsilon)

    def get_next_max_action(self, state):
        return self.model.next_max_action(state)

    def store_experience(self, experience):
        self.model.store_experience(experience)

    def pull_experience_from_buffer(self):
        return self.model.get_experience_from_buffer()

    def check_training_status(self, min_buffer_size):
        return len(self.model.experience_buffer) >= min_buffer_size

    def estimate_q(self, state, action, fixed):
        return self.model.estimate_q(state, action, fixed)

    def train_model(self, gamma):

        state, action, reward, next_state = self.pull_experience_from_buffer()
        next_action = self.get_next_max_action(state)
        q_hat = reward + (gamma *
                          self.estimate_q(next_state, next_action, fixed=True))
        q_current = self.estimate_q(state, action, fixed=False)

        delta_q = q_hat - q_current

        self.update_model_weights(loss=delta_q)

    def update_model_weights(self, loss):
        self.model.update_model_weights(loss)

    @property
    def epsilon(self):
        return self.model.get_epsilon(step_count=self.step_count)

    def training_finished(self, train_config):
        return self.model.terminate_training_status(
            train_config, step_count=self.step_count)

    def store_reward(self, reward):
        self.episode_score += reward

    def record_episode_score(self):

        pass

