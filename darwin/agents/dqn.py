from collections import deque
import numpy as np
import random

import torch
import torch.nn as nn

from utils.util import convert_obs, idx_to_action, action_to_idx


# Discount
GAMMA = 0.99
# Soft update
ALPHA = 0.5
# % chance of applying random action
EPSILON = 0.1

NUM_ITERATIONS = 2000
LEARNING_RATE = 1e-3

# Experience replay hyperparameters
REPLAY_SIZE = 3000
MIN_REPLAY_SIZE = 1000

N_UPDATE_TARGET = 10

TRAINING_EPISODES = 1000


class DQN(nn.Module):
    def __init__(self, n_agents, hidden_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride=(1, 3))

        self.fc_input_size = (n_agents * 7 * 8)
        self.fc1 = nn.Linear(self.fc_input_size, hidden_size)
        self.fc1_bn = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, 12 * 12 * 12)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, self.fc_input_size)
        x = self.relu(self.fc1_bn(self.fc1(x)))
        x = self.fc2(x)
        return x


class DQNAgent:
    def __init__(self, env, training=True, gamma=GAMMA, 
                 batch_size=64, learning_rate=LEARNING_RATE,
                 num_iterations=NUM_ITERATIONS, step_size=1,
                 hidden_size=None, replay_size=REPLAY_SIZE,
                 update_target=10):
        self.env = env
        self.n_agents = self.env.metadata['n_agents']
        self.training = training
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.learning_rate =  learning_rate
        self.step_size = step_size
        self.hidden_size = hidden_size
        self.replay_size = replay_size
        self.update_target = update_target
        self.update_target_cnt = 0

        self.model = DQN(self.n_agents, self.hidden_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.target_model = DQN(self.n_agents, self.hidden_size)

        self.replay = deque(maxlen=self.replay_size)        

    def train(self):
        if len(self.replay) < self.replay_size:
            return

        batch = random.sample(self.replay, self.batch_size)

        curr_states = np.array([convert_obs(tup[0]) for tup in batch])
        curr_q_vectors = self.model(curr_states)

        next_states = np.array([convert_obs(tup[3]) for tup in batch])
        next_q_vectors = self.target_model(next_states)

        train_loss = []
        for i, (curr_state, action, reward, next_state, done) in enumerate(batch):
            if done:
                new_q = reward
            else:
                max_next_q = np.max(next_q_vectors[i])
                new_q = reward + (self.gamma * max_next_q)

            curr_q_vector = curr_q_vectors[i]
            curr_q_vector[action_to_idx(action)] = new_q
            label = curr_q_vector

            self.optimizer.zero_grad()

            pred_q = self.model(curr_state)

            loss = self.loss_fn(pred_q.squeeze(), label.float())

            loss.backward()
            self.optimizer.step()

            train_loss.append(loss.item())

        print(f"Training loss: {np.average(train_loss)}")

        self.update_target_cnt += 1

        if self.update_target_cnt > self.update_target:
            model_state_dict = self.model.state_dict()
            target_model_state_dict = self.target_model.state_dict()
            for name, param in target_model_state_dict.items():
                updated_param = model_state_dict[name]
                target_model_state_dict[name].copy_(updated_param)

    def update_replay(self, experience):
        self.replay.append(experience)   



    # def update(self, s, a, s_next, done):
    #     """
    #     Train the NN during the update step
    #     - Collect trajectory from environment by taking steps
    #     - Use trajectories as training data to train MLP
    #     - Update policy
    #     """
    #     max_q_n = max([self.Q[s_n, a_n] for a_n in actions])
    #     # Don't include next state if done is 1 (if at final state)
    #     self.Q[s, a] += ALPHA * (r + GAMMA * max_q_n * (1 - done) - self.Q[s, a])

    # def act(self, obs):
    #     # Force observation using epsilon-greedy
    #     if np.random.random() < EPSILON:
    #         return self.env.action_space.sample()
        
    #     action_val_pairs = [(a, self.Q[obs, a]) for a in actions]
    #     max_q = max(action_val_pairs, key=lambda x: x[1])
    #     action_choices = [a for a, q in action_val_pairs if q == max_q]
    #     # Select random action of action set with maximum reward value
    #     return np.random.choice(action_choices)
