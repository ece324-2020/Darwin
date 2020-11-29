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
        # Actions are 1x3 matrix, 12 discrete possible values for each element
        # Change so each policy has its own index that is associated with a potential action/observation space
        # Index created whenever policies are created
        # This way if agents have different action spaces we are only choosing from their own
        # I don't want to index it using variable from the MultiDiscrete class like this but don't know
        # what choice we have
        r1, r2, r3 = env.action_space.spaces['action_movement'].spaces[0].nvec
        self.actions = [(x, y, z) for x in range(r1) for y in range(r2) for z in range(r3)]
        self.env = env

    def update(self, s, r, a, s_n, done):
        """
        Args:
        - s: current observation
        - a: current action
        - s_next: next observation
        """
        # print('s_n', s_n)
        s_n_key = tuple(s_n['observation_self'][0])
        max_q_n = max([self.Q[s_n_key, a_n] for a_n in self.actions])
        # Don't include next state if done is 1 (if at final state)
        s_key = tuple(s['observation_self'][0])
        a_key = tuple(a['action_movement'][0])
        self.Q[s_key, a_key] += ALPHA * (r + GAMMA * max_q_n * (1 - done) - self.Q[s_key, a_key])

    def act(self, obs):
        # # Force observation using epsilon-greedy
        if np.random.random() < EPSILON:
            sample = self.env.action_space.sample()
            new_sample = collections.OrderedDict()
            new_sample['action_movement'] = [sample['action_movement'][0]]
            #print("-------------------------Random sample selected---------------------------")
            return map_sample_to_action(sample, is_gym_space=True)

        obs_key = tuple(obs['observation_self'][0])
        action_val_pairs = [(a, self.Q[obs_key, a]) for a in self.actions]
        max_q = max(action_val_pairs, key=lambda x: x[1])[1]
        action_choices = [a for a, q in action_val_pairs if q == max_q]
        # Select random action of action set with maximum reward value
        choice = list(action_choices[np.random.choice(len(action_choices))])
        return map_sample_to_action(choice)

    def reset(self):
        pass


def map_sample_to_action(sample, is_gym_space=False):
    new_sample = collections.OrderedDict()
    if is_gym_space:
        new_sample['action_movement'] = [sample['action_movement'][0]]
    else:
        new_sample['action_movement'] = [np.array(sample)]
    return new_sample
