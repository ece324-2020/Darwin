import gym
from gym.spaces import Dict, Box
import numpy as np
from copy import deepcopy
import logging

BIN_CNT = 11


class DiscretizedObservationWrapper(gym.ObservationWrapper):
    """
    Translate continuous observation space
    """
    def __init__(self, env, bins=BIN_CNT):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)

        # If the env has multiple observation spaces, we will loop through
        # and translate all them to discrete here
        # Similar to mae_envs wrappers/util.py DiscretizedActionWrapper
        self.bin_cnt = bins
        self.low, self.high = env.observation_space.low, env.observation_space.high
        self.discrete_observation_map = np.array([np.linspace(low, high, self.bin_cnt) for low, high in zip(self.low.flatten(), self.high.flatten())])
        self.observation_space = gym.spaces.Discrete(self.bin_cnt ** self.low.flatten().shape[0])

    # def observation(self, obs):
    #     digits = [np.digitize([x], bins)[0] for x, bins in zip(obs.flatten(), self.discrete_observation_map)]
    #     return sum([d * ((self.bin_cnt) ** i) for i, d in self.low.flatten().shape[0]])


    def _convert_to_one_number(self, digits):
        return sum([d * ((self.bin_cnt + 1) ** i) for i, d in enumerate(digits)])

    def observation(self, observation):
        digits = [np.digitize([x], bins)[0]
                  for x, bins in zip(observation.flatten(), self.discrete_observation_map)]
        return self._convert_to_one_number(digits)


class DiscretizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env, action_key, nbuckets=11):
        super().__init__(env)
        self.action_key = action_key
        self.discrete_to_continuous_act_map = []
        for i, ac_space in enumerate(self.action_space.spaces[action_key].spaces):
            assert isinstance(ac_space, Box)
            action_map = np.array([np.linspace(low, high, nbuckets)
                                   for low, high in zip(ac_space.low, ac_space.high)])
            _nbuckets = np.ones((len(action_map))) * nbuckets
            self.action_space.spaces[action_key].spaces[i] = gym.spaces.MultiDiscrete(_nbuckets)
            self.discrete_to_continuous_act_map.append(action_map)
        self.discrete_to_continuous_act_map = np.array(self.discrete_to_continuous_act_map)

    def action(self, action):
        action = deepcopy(action)
        ac = action[self.action_key]
        print(ac)

        # helper variables for indexing the discrete-to-continuous action map
        agent_idxs = np.tile(np.arange(ac.shape[0])[:, None], ac.shape[1])
        ac_idxs = np.tile(np.arange(ac.shape[1]), ac.shape[0]).reshape(ac.shape)

        # print(self.action_key)
        # print(agent_idxs)
        # print(ac_idxs)
        # print(ac)
        # print(action)
        # print(self.discrete_to_continuous_act_map)

        action[self.action_key] = self.discrete_to_continuous_act_map[agent_idxs, ac_idxs, ac]
        return action

def update_obs_space(env, delta):
    spaces = env.observation_space.spaces.copy()
    for key, shape in delta.items():
        spaces[key] = Box(-np.inf, np.inf, shape, np.float32)
    return Dict(spaces)

