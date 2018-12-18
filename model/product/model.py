"""
Product model solution for the banana navigation problem
"""
import os
import sys
import json
import numpy as np
from src.base_model.base_model import BaseModel
from src.base_networks.base_network import Network
import torch
nn = torch.nn

WORK_DIR = os.environ['ROOT_DIR']
sys.path.append(WORK_DIR)


class Model(BaseModel):
    def __init__(self, model_name, experiment_id, nb_state_features, nb_actions,
                 hyperparams, overwrite_experiment):
        super(Model, self).__init__(
            model_name=model_name,
            experiment_id=experiment_id,
            hyperparams=hyperparams,
            overwrite_experiment=overwrite_experiment)

        with open(os.path.join(WORK_DIR, 'model', model_name, experiment_id,
                               "params.json")) as handle:
            self.params = json.load(handle)
        self.nb_actions = nb_actions
        self.state_size = nb_state_features

        self.experience_buffer = []

        self.network = Network(
            nb_features=self.state_size, nb_actions=self.nb_actions,
            params=self.params, seed=self.hyperparams['random_seed'])
        self.fixed_network = Network(
            nb_features=self.state_size, nb_actions=self.nb_actions,
            params=self.params, seed=self.hyperparams['random_seed'])

        self.soft_update(src_model=self.network, dst_model=self.fixed_network,
                         tau=1.0)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            params=self.network.parameters(),
            lr=self.hyperparams['init_learning_rate'])

    def next_action(self, state, epsilon):
        state_tensor = torch.from_numpy(state).float()

        action_values = self.network.network.forward(state_tensor).data.numpy()
        action = np.argmax(action_values, axis=-1)

        if np.random.rand() < epsilon:
            action = np.random.randint(0, self.nb_actions)

        return action

    def next_max_action(self, state):
        state_tensor = torch.from_numpy(state).float()

        action_values = self.network.network.forward(
            state_tensor).detach().data.numpy()
        action = np.argmax(action_values, axis=-1)

        return action

    def store_experience(self, experience):
        self.experience_buffer.append(experience)

    def get_sarsa(self):

        experience_index = np.random.choice(
            range(len(self.experience_buffer)), self.hyperparams['batch_size'],
            replace=False)

        experiences = [self.experience_buffer.pop(index) for index in sorted(
            experience_index, reverse=True)]

        states, actions, rewards, next_states = zip(*experiences)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)

        return states, actions, rewards, next_states

    def train_model(self, states, actions, rewards, next_states, next_actions):
        gamma = self.hyperparams['gamma']

        state_values = self.estimate_action_values(states=next_states,
                                                   fixed=True).detach()
        action_values = state_values.max(dim=1)[0]
        q_targets = rewards + (gamma * action_values)

        self.optimizer.zero_grad()
        expected_state_values = self.estimate_action_values(states=states,
                                                            fixed=False)

        q_expected = expected_state_values[range(0, len(actions)), actions]

        q_delta = self.criterion(q_expected, q_targets.float())
        q_delta.backward()
        self.optimizer.step()

        self.soft_update(src_model=self.network.network,
                         dst_model=self.fixed_network.network,
                         tau=self.params['tau'])

    def estimate_action_values(self, states, fixed):
        states = torch.from_numpy(states).float()
        if fixed:
            action_values_tensor = self.fixed_network.forward(states)
        else:
            action_values_tensor = self.network.forward(states)

        return action_values_tensor

    def soft_update(self, src_model, dst_model, tau):
        for dst_param, src_param in zip(dst_model.parameters(),
                                        src_model.parameters()):
            updated_param = tau*src_param.data + (1.0-tau)*dst_param.data
            dst_param.data.copy_(updated_param)

    def get_epsilon(self, step_count):
        epf = self.hyperparams['epsilon_root_factor']
        if step_count == 0:
            epsilon = 1.0
        else:
            epsilon = (1.0/step_count)**(1.0/epf)
        return np.round(epsilon, 3)

    def terminate_training_status(self, episode_count, **kwargs):
        return episode_count >= self.hyperparams['max_episodes']

    def checkpoint_model(self, episode_count):
        checkpoint_filename = os.path.join(
            self.checkpoint_dir, 'ckpt_{}.pth'.format(episode_count))
        torch.save(self.network.network.state_dict(), checkpoint_filename)

    def restore_checkpoint(self, checkpoint):
        checkpoint_fn = os.path.join(self.checkpoint_dir, checkpoint)
        state_dict = torch.load(checkpoint_fn)
        self.network.network.load_state_dict(state_dict)

    def check_training_status(self):
        status = (len(self.experience_buffer) >=
                  self.params['min_buffer_size'])
        return status
