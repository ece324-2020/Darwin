from collections import deque
import numpy as np
import random

from gym.spaces import Dict

import torch
import torch.nn as nn

from utils.util import convert_obs, idx_to_action, action_to_idx


# Discount
GAMMA = 0.99
# Soft update
ALPHA = 0.5
# % chance of applying random action
EPSILON = 0.1

LEARNING_RATE = 1e-3

# Experience replay hyperparameters
REPLAY_CACHE_SIZE = 300
MIN_REPLAY_CACHE_SIZE = 100

N_UPDATE_TARGET = 10

TRAINING_EPISODES = 1000


class DQN(nn.Module):
    def __init__(self, n_agents, hidden_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride=(1, 3))

        self.fc_input_size = (n_agents * 6 * 8)
        self.fc1 = nn.Linear(self.fc_input_size, hidden_size)
        self.fc1_bn = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, 11 * 11 * 11)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, self.fc_input_size)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DQNAgent:
    def __init__(self,
                 env,
                 training=True,
                 gamma=GAMMA,
                 epsilon=EPSILON,
                 batch_size=64, 
                 learning_rate=LEARNING_RATE, 
                 hidden_size=1024, 
                 replay_cache_size=REPLAY_CACHE_SIZE, 
                 min_replay_cache_size=MIN_REPLAY_CACHE_SIZE,
                 update_target_every=10
                 ):
        self.env = env
        self.n_agents = self.env.metadata['n_agents']
        self.training = training
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.learning_rate =  learning_rate
        self.hidden_size = hidden_size
        self.replay_cache_size = replay_cache_size
        self.min_replay_cache_size = min_replay_cache_size
        self.update_target_every = update_target_every

        self.model = DQN(self.n_agents, self.hidden_size)
        self.model = self.model.double()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.target_model = DQN(self.n_agents, self.hidden_size)
        self.target_model = self.target_model.double()

        self.replay_cache = deque(maxlen=self.replay_cache_size)        

    def train(self, step, agent_id):
        if len(self.replay_cache) < self.min_replay_cache_size:
            return

        batch = random.sample(self.replay_cache, self.batch_size)

        curr_states = np.array([convert_obs(tup[0]) for tup in batch])
        curr_states = torch.from_numpy(curr_states)
        curr_q_vectors = self.model(curr_states).detach().numpy()

        next_states = np.array([convert_obs(tup[3]) for tup in batch])
        next_states = torch.from_numpy(next_states)
        next_q_vectors = self.target_model(next_states).detach().numpy()

        train_data = []
        train_labels = []
        for i, (curr_state, action, reward, next_state, done) in enumerate(batch):
            if done:
                new_q = reward
            else:
                max_next_q = np.max(next_q_vectors[i])
                new_q = reward + (self.gamma * max_next_q)

            curr_q_vector = curr_q_vectors[i]
            curr_q_vector[action_to_idx(action)] = new_q
            label = curr_q_vector

            train_data.append(convert_obs(curr_state))
            train_labels.append(label)

        train_data = torch.from_numpy(np.array(train_data))
        train_labels = torch.from_numpy(np.array(train_labels))

        self.optimizer.zero_grad()

        output = self.model(train_data.double())

        loss = self.loss_fn(output.squeeze(), train_labels.double())

        loss.backward()
        self.optimizer.step()

        print(f"Agent {agent_id} - Training Loss: {loss.item()}")

        if step % self.update_target_every == 0:
            model_state_dict = self.model.state_dict()
            target_model_state_dict = self.target_model.state_dict()
            for name, _ in target_model_state_dict.items():
                updated_param = model_state_dict[name]
                target_model_state_dict[name].copy_(updated_param)

    def update_replay_cache(self, experience):
        self.replay_cache.append(experience)   

    def act(self, obs, train=True):
        if train:
            # Force observation using epsilon-greedy
            if np.random.random() < self.epsilon:
                random_action = self.env.action_space.sample()['action_movement'][0]
                random_action = {'action_movement': [random_action]}
                return random_action
        
        obs = convert_obs(obs, eval=True)
        obs = torch.from_numpy(obs)
        with torch.no_grad():
            q_vector = self.model(obs.double())
        best_action_idx = np.argmax(np.array(q_vector))
        best_action = idx_to_action(best_action_idx)
        action = {'action_movement': [best_action]}
        return action
        