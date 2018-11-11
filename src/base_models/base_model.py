"""
Model object that abstracts dynamic model versioning and directory management.
"""

from pydoc import locate


class BaseModel(object):
    def __init__(self, model_name, experiment_id):
        self.model = self.load_model(model_name, experiment_id)
        self.nb_actions = 4
        self.state_shape = 37

    def load_model(self, model_name, experiment_id):
        Model = locate('model.{}.model.Model'.format(model_name))
        model = Model(model_name, experiment_id)
        return model

    def get_next_action(self, state):
        return self.model.next_action(state=state)
