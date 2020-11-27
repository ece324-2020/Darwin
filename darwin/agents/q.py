import collections

import numpy as np


# Discount
GAMMA = 0.99
# Soft update
ALPHA = 0.5
# % chance of applying random action
EPSILON = 0.1

class QAgent:
    def __init__(self, env):
        self.Q = collections.defaultdict(float)
        self.actions = range(len(env.action_space.spaces.items()))
        self.env = env

    def update(self, s, a, s_next, done):
        """
        Args:
        - s: current observation
        - a: current action
        - s_next: next observation
        """
        max_q_n = max([self.Q[s_n, a_n] for a_n in actions])
        # Don't include next state if done is 1 (if at final state)
        self.Q[s, a] += ALPHA * (r + GAMMA * max_q_n * (1 - done) - self.Q[s, a])

    def act(self, obs):
        return self.env.action_space.sample()
        # # Force observation using epsilon-greedy
        # if np.random.random() < EPSILON:
        #     return self.env.action_space.sample()
        
        # # print('obs', obs)
        # # print('0-----------------')
        # # for a in self.actions:
        # #     print(a)

        # action_val_pairs = [(a, self.Q[obs, a]) for a in self.actions]
        # max_q = max(action_val_pairs, key=lambda x: x[1])
        # action_choices = [a for a, q in action_val_pairs if q == max_q]
        # # Select random action of action set with maximum reward value
        # return np.random.choice(action_choices)

    def reset(self):
        pass
