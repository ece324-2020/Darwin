from collections import deque
import numpy as np
import random
from operator import itemgetter

from gym.spaces import Dict

import torch
import torch.nn as nn

from utils.util import listdict2dictnp, split_obs, convert_obs, idx_to_action, action_to_idx


# Discount
GAMMA = 0.99
# Soft update
ALPHA = 0.5
# % chance of applying random action
EPSILON = 0.3

LEARNING_RATE = 1e-3

# Experience replay hyperparameters
REPLAY_CACHE_SIZE = 3000
MIN_REPLAY_CACHE_SIZE = 1000
N_UPDATE_TARGET = 5000

TRAINING_EPISODES = 1000


class DqnConv(nn.Module):
    def __init__(self, n_agents, hidden_size):
        super(DqnConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, (n_agents+1), stride=(1, (n_agents+1)))

        self.fc_input_size = (n_agents * (8-n_agents) * 8)
        self.fc1 = nn.Linear(self.fc_input_size, hidden_size)
        self.fc1_bn = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, int(0.5 * hidden_size))
        self.fc2_bn = nn.BatchNorm1d(int(0.5 * hidden_size))
        self.fc3 = nn.Linear(int(0.5 * hidden_size), int(0.25 * hidden_size))
        self.fc4 = nn.Linear(int(0.25 * hidden_size), 2 * 2 * 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, self.fc_input_size)
        x = self.relu(self.fc1_bn(self.fc1(x)))
        x = self.relu(self.fc2_bn(self.fc2(x)))
        # x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DqnLinear(nn.Module):
    def __init__(self, hidden_size):
        super(DqnLinear, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(8 * 3, hidden_size)
        self.fc1_bn = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2 * hidden_size)
        self.fc2_bn = nn.LayerNorm(2 * hidden_size)
        self.fc3 = nn.Linear(2 * hidden_size, 4 * hidden_size)
        self.fc3_bn = nn.LayerNorm(4 * hidden_size)
        self.fc4 = nn.Linear(4 * hidden_size, 8 * hidden_size)
        self.fc5 = nn.Linear(8 * hidden_size, 11 * 11 * 11)
        self.relu = nn.ReLU()

    def forward(self, x):
      
        x = self.relu(self.fc1_bn(self.fc1(x)))
        x = self.relu(self.fc2_bn(self.fc2(x)))
        x = self.relu(self.fc3_bn(self.fc3(x)))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class DQNAgent:
    def __init__(self,
                 env,
                 training=True,
                 model_type='cnn',
                 gamma=GAMMA,
                 epsilon=EPSILON,
                 batch_size=64, 
                 learning_rate=LEARNING_RATE, 
                 hidden_size=256, 
                 replay_cache_size=REPLAY_CACHE_SIZE, 
                 min_replay_cache_size=MIN_REPLAY_CACHE_SIZE,
                 update_target_every=N_UPDATE_TARGET,
                 policy_name = "baseline",
                 model = None
                 ):
        self.env = env
        self.n_agents = self.env.metadata['n_agents']
        self.training = training
        self.model_type = model_type
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.learning_rate =  learning_rate
        self.hidden_size = hidden_size
        self.replay_cache_size = replay_cache_size
        self.min_replay_cache_size = min_replay_cache_size
        self.update_target_every = update_target_every
        self.policy_name = policy_name
        self.model = model
        self.update_target_count = 1

        if self.model_type == 'cnn':
            print("cnn")
            if (self.model == None):
                self.model = DqnConv(self.n_agents, 2 * self.hidden_size)
                self.model = self.model.double()
                self.target_model = DqnConv(self.n_agents, 2 * self.hidden_size)
                self.target_model = self.target_model.double()
            
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.loss_fn = nn.MSELoss()
            
            

        elif self.model_type == 'linear':
            self.model = DqnLinear(self.hidden_size)
            self.model = self.model.double()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.loss_fn = nn.MSELoss()

            self.target_model = DqnLinear(self.hidden_size)
            self.target_model = self.target_model.double()


        self.replay_cache = deque(maxlen=self.replay_cache_size)        

  

        
      
    def train(self, step, agent_id):
        if len(self.replay_cache) < self.min_replay_cache_size:
            return

        batch = random.sample(self.replay_cache, self.batch_size)

        curr_states = np.array([convert_obs(tup[0], model_type=self.model_type) for tup in batch])
        curr_states = torch.from_numpy(curr_states)
       

        curr_q_vectors = self.model(curr_states).detach().numpy()
        next_states = np.array([convert_obs(tup[3], model_type=self.model_type) for tup in batch])
        next_states = torch.from_numpy(next_states)
        next_q_vectors = self.target_model(next_states).detach().numpy()
      
        next_max_q = self.target_model(next_states).detach().max(1)[0]
 
        train_data = []
        train_labels = []
        for i, (curr_state, action, reward, next_state, done) in enumerate(batch):
            
            '''
            if done:
                new_q = reward
            else:
                max_next_q = np.max(next_q_vectors[i])
                new_q_old = reward + (self.gamma * max_next_q)
            '''
            if done:
                new_q = reward
            else:
                new_q = reward + (self.gamma * next_max_q[i])
            
            curr_q_vector = curr_q_vectors[i]
            if (self.model_type == "linear"):
                curr_q_vector = curr_q_vector[0]
           
            curr_q_vector[action_to_idx(action)] = new_q
            label = curr_q_vector
            # label = np.ones(curr_q_vector.shape) * float(new_q)
  
            train_data.append(convert_obs(curr_state, model_type=self.model_type))
            train_labels.append(label)
            

        train_data = torch.from_numpy(np.array(train_data))
        train_labels = torch.from_numpy(np.array(train_labels))

        self.optimizer.zero_grad()

        output = self.model(train_data.double())

        loss = self.loss_fn(output.squeeze(), train_labels.double())
        loss.backward()
        self.optimizer.step()

        print(f"Agent {agent_id} - Training Loss: {loss.item()}")

        # if step % self.update_target_every == 0:
        '''
        if step == 0:
            model_state_dict = self.model.state_dict()
            target_model_state_dict = self.target_model.state_dict()
            for name, _ in target_model_state_dict.items():
                updated_param = model_state_dict[name]
                target_model_state_dict[name].copy_(updated_param)
        '''
        if self.update_target_count == self.update_target_every:
            print("--------------------------------------------------------")
            print()
            print()
            print("                UPDATED TARGET NETWORK                  ")
            print()
            print()
            print("--------------------------------------------------------")
            model_state_dict = self.model.state_dict()
            target_model_state_dict = self.target_model.state_dict()
            for name, _ in target_model_state_dict.items():
                updated_param = model_state_dict[name]
                target_model_state_dict[name].copy_(updated_param)

            self.update_target_count = 1
        return loss.item()
    def update_replay_cache(self, experience, agent_id):
        if self.model_type == 'linear':
            s, a, r, s_next, done = experience
            new_s = self.individual_obs(s, agent_id=agent_id)
            new_s_next = self.individual_obs(s_next, agent_id=agent_id)
            self.replay_cache.append((new_s, a, r, new_s_next, done))
        else:
            self.replay_cache.append(experience)

    def save_policy(self, agent_id, policy_name='sleep', model_type='linear'):
        #torch.save(self.model, f"dqn_{policy_name}_{model_type}_agent{agent_id}.pt")
        torch.save(self.model,f"final_dqn_sleep_cnn_agent{agent_id}.pt")

    def individual_obs(self, obs, agent_id):
        full_obs = split_obs(obs, keepdims=False)
        idx = np.split(np.arange(len(full_obs)), self.n_agents)
       
        ob = itemgetter(*idx[agent_id])(full_obs)
   
        ob = listdict2dictnp([ob] if idx[agent_id].shape[0] == 1 else ob)
        return ob

    def act(self, obs, agent_id, train=True,model=None):
        if train:
            # Force observation using epsilon-greedy
            if np.random.random() < self.epsilon:
                
                random_action = self.env.action_space.sample()['action_movement'][0]
                random_action = {'action_movement': [random_action]}
                
                return random_action

        if (model != None):
            self.model = model

        if self.model_type == 'linear':
            obs = self.individual_obs(obs, agent_id=agent_id)
        
    
        obs = convert_obs(obs, model_type=self.model_type, eval=True)
        obs = torch.from_numpy(obs)
        
        
        self.model.eval()
        with torch.no_grad():
    
            q_vector = self.model(obs.double())
        
        self.model.train()
        
        best_action_idx = np.argmax(np.array(q_vector))
        best_action = idx_to_action(best_action_idx)
        action = {'action_movement': [best_action]}
        
        return action
        