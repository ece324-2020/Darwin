import gym
import numpy as np
from gym.spaces import Tuple, MultiDiscrete, Dict, Box
import math


def update_obs_space(env, delta):
    spaces = env.observation_space.spaces.copy()
    for key, shape in delta.items():
        spaces[key] = Box(-np.inf, np.inf, shape, np.float32)
    return Dict(spaces)

def reward_function(x):
    return 2/(math.exp(5*x) + math.exp(-5*x))


class AgentHealthWrapper(gym.Wrapper):
    '''
    Add agent health to underlying env.
    '''
    def __init__(self, env, max_agent_health=100, starting_health_discount=0.75, agent_health_tick=1, reward_scale=1.0, reward_scale_obs=False):
        super().__init__(env)
        self.env = env
        self.n_agents = self.env.metadata['n_agents']
        self.max_agent_health = max_agent_health
        self.agent_health_tick = agent_health_tick
        self.reward_scale = reward_scale
        self.reward_scale_obs = reward_scale_obs
        self.starting_health_discount = starting_health_discount
        
        if type(reward_scale) not in [list, tuple, np.ndarray]:
            self.reward_scale = [reward_scale, reward_scale]

        self.env.metadata['agent_health_tick'] = agent_health_tick
        self.env.metadata['max_agent_health'] = max_agent_health

        self.observation_space = update_obs_space(self.env, {'agent_health': (self.max_agent_health, 1)})

    def reset(self):
        obs = self.env.reset()
        sim = self.unwrapped.sim

        # Reset agent healths
        self.agent_healths = np.ones((self.n_agents, 1)) * self.max_agent_health * self.starting_health_discount
        self.curr_reward_scale = np.random.uniform(self.reward_scale[0], self.reward_scale[1])
        return self.observation(obs)

    def observation(self, obs):
        # Add agent health to observations
        obs['agent_health'] = self.agent_healths
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        sleep_healths = obs['agent_sleep_health']
        food_healths = obs['agent_food_health']

        health_rew = np.zeros((self.n_agents,))
        mid = self.env.metadata['max_sleep_health'] + self.env.metadata['max_food_health']
        # self.agent_healths = obs['agent_health']
        for agent in range(self.n_agents):
            self.agent_healths[agent] = sleep_healths[agent] + food_healths[agent]
            #print('agent sleep health', sleep_healths[agent])
            #print('agent food health', food_healths[agent])
            #print('agent health', self.agent_healths[agent])
            # Check agent health for reward
            if self.agent_healths[agent] <= 25:
                health_rew[agent] -= 1
            elif self.agent_healths[agent] <= 50:
                health_rew[agent] -= 0.5
            elif 50 <= self.agent_healths[agent] <= self.max_agent_health:
                health_rew[agent] += 1

            # Agent health decreases with each time step
            # self.agent_healths[agent] -= self.agent_health_tick

        info['agent_health'] = self.agent_healths
        rew += health_rew * self.curr_reward_scale

        for h in self.observation(obs)['agent_health']:
            if h <= 0.:
                done = True

        return self.observation(obs), rew, done, info

    
class SleepHealthWrapper(gym.Wrapper):
    '''
    Add sleep health to underlying env.
    '''
    def __init__(self, env, n_sites=2, site_thresh=0.5, max_sleep=50, starting_discount=0.8, sleep_replenish_tick=1, loss_replenish_factor=0.25, reward_scale=1.0, reward_scale_obs=False):
        super().__init__(env)
        self.env = env
        self.n_sites = n_sites
        self.site_thresh = site_thresh
        self.max_sleep = max_sleep
        self.reward_scale = reward_scale
        self.reward_scale_obs = reward_scale_obs
        self.starting_discount = starting_discount
        self.sleep_replenish_tick = sleep_replenish_tick
        self.loss_replenish_factor = loss_replenish_factor
        self.sleep_loss_tick = sleep_replenish_tick * loss_replenish_factor

        self.n_agents = self.env.metadata['n_agents']

        if type(reward_scale) not in [list, tuple, np.ndarray]:
            self.reward_scale = [reward_scale, reward_scale]

        self.env.metadata['sleep_replenish_tick'] = sleep_replenish_tick
        self.env.metadata['sleep_loss_tick'] = self.sleep_loss_tick
        self.env.metadata['max_sleep_health'] = max_sleep
        self.env.metadata['n_sleep_sites'] = self.n_sites

        sleep_dim = 5 if self.reward_scale_obs else 4
        self.observation_space = update_obs_space(self.env, {'agent_sleep_health': (self.max_sleep, 1),
                                                             'sleep_obs': (self.n_sites, sleep_dim)})
        self.action_space.spaces['action_sleep'] = Tuple([MultiDiscrete([2] * self.n_sites)
                                                        for _ in range(self.n_agents)])

    def reset(self):
        obs = self.env.reset()
        sim = self.unwrapped.sim

        # Reset agent sleep healths
        self.sleep_healths = np.ones((self.n_agents, 1)) * self.max_sleep * self.starting_discount
        self.curr_reward_scale = np.random.uniform(self.reward_scale[0], self.reward_scale[1])
        return self.observation(obs)

    def observation(self, obs):
        # Add agent sleep to observations
        sleep_pos = obs['sleep_pos']
        obs['agent_sleep_health'] = self.sleep_healths
        obs['sleep_obs'] = np.concatenate([sleep_pos, self.sleep_healths], 1)
        return obs

    def step(self, action):
        action_sleep = action.pop('action_sleep')
        obs, rew, done, info = self.env.step(action)

        if self.n_sites > 0:
            dist_to_site = np.linalg.norm(obs['agent_pos'][:, None] - obs['sleep_pos'][None], axis=-1)
            sleep = np.logical_and(dist_to_site < self.site_thresh, action_sleep).astype(np.float32)
            for agent in range(self.n_agents):
                agent_on_sleep_site = False
                for sleep_site in range(self.n_sites):
                    # Agent is on sleep site
                    if sleep[agent][sleep_site] == 1:
                        # Update sleep health with limit as max sleep
                        new_health = self.sleep_healths[agent] + self.sleep_replenish_tick
                        self.sleep_healths[agent] = min(self.max_sleep, new_health)
                        agent_on_sleep_site = True
                
                if not agent_on_sleep_site:
                    # Agent sleep health decreases
                    self.sleep_healths[agent] -= self.sleep_loss_tick

        info['agent_sleep_health'] = self.sleep_healths
        
        for h in self.observation(obs)['agent_sleep_health']:
            if h <= 0.:
                done = True

        return self.observation(obs), rew, done, info


class AlwaysSleepWrapper(gym.ActionWrapper):
    '''
        Remove sleep action and replace it with always sleeping.
        Args:
            agent_idx_allowed (ndarray): indicies of agents allowed to sleep.
    '''
    def __init__(self, env, agent_idx_allowed):
        super().__init__(env)
        self.action_space.spaces.pop('action_sleep')
        self.agent_idx_allowed = agent_idx_allowed

    def action(self, action):
        action['action_sleep'] = np.zeros((self.metadata['n_agents'], self.metadata['n_sleep_sites']))
        action['action_sleep'][self.agent_idx_allowed] = 1.
        return action


class AltFoodHealthWrapper(gym.Wrapper):
    '''
        Adds food health to underlying env.
        Manages food levels.

        Args:
            eat_thresh (float): radius within which food items can be eaten
            max_food_health (int): number of times a food item can be eaten
                                   before it disappears
            respawn_time (int): Number of time steps after which food items
                                that have been eaten reappear
            reward_scale (float or (float, float)): scales the reward by this amount. If tuple of
                floats, the exact reward scaling is uniformly sampled from
                (reward_scale[0], reward_scale[1]) at the beginning of every episode.
            reward_scale_obs (bool): If true, adds the reward scale for the current
                episode to food_obs
    '''
    def __init__(self, env, n_food=10,eat_thresh=0.5, max_food_health=5, agent_max_food=50, agent_starting_discount=0.6, agent_food_loss_tick=0.25, respawn_time=40,
                 reward_scale=1.0, reward_scale_obs=False):
        super().__init__(env)
        self.n_food = n_food
        self.env = env
        self.eat_thresh = eat_thresh
        self.max_food_health = max_food_health
        self.respawn_time = respawn_time
        self.on_reset_step = False
        self.reward_scale = reward_scale
        self.reward_scale_obs = reward_scale_obs
        self.agent_max_food = agent_max_food
        self.agent_starting_discount = agent_starting_discount
        self.agent_food_loss_tick = agent_food_loss_tick

        self.n_agents = self.env.metadata['n_agents']

        if type(reward_scale) not in [list, tuple, np.ndarray]:
            self.reward_scale = [reward_scale, reward_scale]

        self.env.metadata['max_food_health'] = agent_max_food

        # Reset obs/action space to match
        self.max_n_food = self.metadata['max_n_food']
        self.curr_n_food = self.metadata['curr_n_food']
        self.max_food_size = self.metadata['food_size']
        food_dim = 5 if self.reward_scale_obs else 4
        self.observation_space = update_obs_space(self.env, {'food_obs': (self.max_n_food + self.n_agents, food_dim),
                                                             'food_health': (self.max_n_food, 1),
                                                             'food_eat': (self.max_n_food, 1),
                                                             'agent_food_health': (self.agent_max_food, 1)})
        self.action_space.spaces['action_eat_food'] = Tuple([MultiDiscrete([2] * self.max_n_food)
                                                             for _ in range(self.n_agents)])

    def reset(self):
        obs = self.env.reset()
        sim = self.unwrapped.sim

        # Reset obs/action space to match
        self.curr_n_food = self.metadata['curr_n_food']

        self.agent_food_healths = np.ones((self.n_agents, 1)) * self.agent_max_food * self.agent_starting_discount
        self.food_site_ids = np.array([sim.model.site_name2id(f'food{i}')
                                       for i in range(self.curr_n_food)])
        # Reset food healths
        self.food_healths = np.ones((self.curr_n_food, 1)) * self.max_food_health
        self.eat_per_food = np.zeros((self.curr_n_food, 1))

        # Reset food size
        self.respawn_counters = np.zeros((self.curr_n_food,))
        self.on_reset_step = True

        self.curr_reward_scale = np.random.uniform(self.reward_scale[0], self.reward_scale[1])
        return self.observation(obs)

    def observation(self, obs):
        # Add food position and healths to observations
        food_pos = obs['food_pos']
        obs['food_health'] = self.food_healths
        obs['agent_food_health'] = self.agent_food_healths
        obs['food_obs'] = np.concatenate([food_pos, self.food_healths], 1)
        if self.reward_scale_obs:
            obs['food_obs'] = np.concatenate([obs['food_obs'], np.ones((self.curr_n_food, 1)) * self.curr_reward_scale], 1)
        obs['food_eat'] = self.eat_per_food
        return obs

    def step(self, action):
        action_eat_food = action.pop('action_eat_food')
        obs, rew, done, info = self.env.step(action)

        if self.curr_n_food > 0:
            # Eat food that is close enough
            dist_to_food = np.linalg.norm(obs['agent_pos'][:, None] - obs['food_pos'][None], axis=-1)
            #print(f"max_dist_to_food:{max(max(dist_to_food[0]),max(dist_to_food[1]))}\n")
            eat = np.logical_and(dist_to_food < self.eat_thresh, self.food_healths.T > 0)
            #print(f"eat_before:{eat}")
            eat = np.logical_and(eat, action_eat_food).astype(np.float32)
            #print(f"food_heath:{self.food_healths}")
            for agent in range(self.n_agents):
                agent_on_food_site = False
                for agent_food in range(self.n_food):
                    if eat[agent][agent_food] == 1:
                        agent_on_food_site = True if self.agent_food_healths[agent] < self.agent_max_food else False
                        eat[agent][agent_food] = reward_function(dist_to_food[agent][agent_food]) * self.max_food_health
                        #print(eat[agent][agent_food])
                        if eat[agent][agent_food] > self.food_healths[agent_food][0]:
                            eat[agent][agent_food] = self.food_healths[agent_food][0] 
                            if (agent == 0):
                                eat[1][agent_food] = 0

                        # Assert agent health is only ever as large as max health
                        self.agent_food_healths[agent] = min(self.agent_food_healths[agent], self.agent_max_food)

                        new_agent_fh = eat[agent][agent_food] + self.agent_food_healths[agent]
                        if new_agent_fh > self.agent_max_food:
                            # Food eaten will only bring agent health up to maximum, cannot over eat
                            eat[agent][agent_food] = self.agent_max_food - self.agent_food_healths[agent]
                            self.agent_food_healths[agent] = self.agent_max_food
                        else:
                            self.agent_food_healths[agent] = new_agent_fh
                
                # Agent food health decreases
                if not agent_on_food_site:
                    self.agent_food_healths[agent] -= self.agent_food_loss_tick

            
            eat_per_food = np.sum(eat, 0)
           
            #print(f"eat_per_food:{eat_per_food}\n")
            # Make sure that all agents can't have the last bite of food.
            # At that point, food is split evenly
            over_eat = self.food_healths[:, 0] < eat_per_food
            eat[:, over_eat] *= (self.food_healths[over_eat, 0] / eat_per_food[over_eat])
            eat_per_food = np.sum(eat, 0)
            self.eat_per_food = eat_per_food[:, None]
            #print(f"eat_per_food:{self.eat_per_food}")
            # Update food healths and sizes
            self.food_healths -= eat_per_food[:, None]
            health_diff = eat_per_food[:, None]
            size_diff = health_diff * (self.max_food_size / self.max_food_health)
            size = self.unwrapped.sim.model.site_size[self.food_site_ids] - size_diff
            size = np.maximum(0, size)
            self.unwrapped.sim.model.site_size[self.food_site_ids] = size

            if not self.respawn_time == np.inf:
                self.food_healths[self.respawn_counters == self.respawn_time] = self.max_food_health
                self.unwrapped.sim.model.site_size[self.food_site_ids[self.respawn_counters == self.respawn_time]] = self.max_food_size
            elif self.on_reset_step:
                self.food_healths[:] = self.max_food_health
                self.unwrapped.sim.model.site_size[self.food_site_ids[:]] = self.max_food_size
                self.on_reset_step = False

            self.respawn_counters[self.food_healths[:, 0] == 0] += 1
            self.respawn_counters[self.food_healths[:, 0] != 0] = 0
            for i in range(self.n_food):
                if (self.food_healths[i][0] < 0.):
                    self.food_healths[i][0] = 0.
            assert np.all(self.food_healths >= 0.), \
                f"There is a food health below 0: {self.food_healths}"

            # calculate food reward
            # food_rew = np.sum(eat, axis=1)

        info['agents_eat'] = eat
        # print("food health: ", self.observation(obs)['food_health'])
        # done = True
        # for h in self.observation(obs)['food_health']:
        #     if h[0] > 0.:
        #         done = False
        done = False
        for h in self.observation(obs)['agent_food_health']:
            if h <= 0.:
                done = True

        return self.observation(obs), rew, done, info

