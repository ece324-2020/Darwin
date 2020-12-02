import numpy as np
import gym
from gym.spaces import Box, Dict

from mujoco_worldgen import Floor, WorldBuilder, Geom, ObjFromXML, WorldParams, Env
from modules.wall import RandomWalls,WallScenarios,Wall

from wrappers.food import FoodHealthWrapper, AlwaysEatWrapper
from wrappers.multi_agent import (SplitMultiAgentActions, SplitObservations,
                                    SelectKeysWrapper)
from wrappers.lidar import Lidar
from wrappers.discrete import DiscretizeActionWrapper
from modules.food import Food
from modules.agents import Agents
from modules.util import uniform_placement
from modules.lidar import LidarSites


class TrackStatWrapper(gym.Wrapper):
    def __init__(self, env, n_food):
        super().__init__(env)
        self.n_food = n_food

    def reset(self):
        obs = self.env.reset()
        if self.n_food > 0:
            self.total_food_eaten = np.sum(obs['food_eat'])

            self.in_prep_phase = True

        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        if self.n_food > 0:
            self.total_food_eaten += np.sum(obs['food_eat'])

        if self.in_prep_phase and obs['prep_obs'][0, 0] == 1.0:
            self.in_prep_phase = False

            if self.n_food > 0:
                self.total_food_eaten_prep = self.total_food_eaten

        if done:
            if self.n_food > 0:
                info.update({
                    'food_eaten': self.total_food_eaten,
                    'food_eaten_prep': self.total_food_eaten_prep
                })
        
        return obs, rew, done, info


class BaselineRewardWrapper(gym.Wrapper):
    def __init__(self, env, n_agents):
        super().__init__(env)
        self.n_agents = self.unwrapped.n_agents
        
        self.metadata['n_agents'] = self.n_agents

        self.unwrapped.agent_names = [f'agent{i}' for i in range(self.n_agents)]

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        this_rew = np.subtract(np.ones((self.n_agents,)), 1.01)
        
        rew += this_rew
        return obs, rew, done, info




def update_obs_space(env, delta):
    spaces = env.observation_space.spaces.copy()
    for key, shape in delta.items():
        spaces[key] = Box(-np.inf, np.inf, shape, np.float32)
    return Dict(spaces)


def rand_pos_on_floor(sim, n=1):
    world_size = sim.model.geom_size[sim.model.geom_name2id('floor0')] * 2
    new_pos = np.random.uniform(np.array([[0.2, 0.2] for _ in range(n)]),
                                np.array([world_size[:2] - 0.2 for _ in range(n)]))
    return new_pos


class Baseline(Env):
    def __init__(self, n_agents=2, n_food=10, horizon=200, n_substeps=10,
                 floor_size=4., grid_size=30, deterministic_mode=False):
        super().__init__(get_sim=self._get_sim,
                         get_obs=self._get_obs,
                         action_space=tuple((-1.0, 1.0)),
                         horizon=horizon,
                         deterministic_mode=deterministic_mode)
        self.n_agents = n_agents
        self.metadata = {}
        self.metadata['n_agents'] = n_agents
        self.n_food = n_food
        self.horizon = horizon
        self.n_substeps = n_substeps
        self.floor_size = floor_size
        self.placement_grid = np.zeros((grid_size, grid_size))
        self.modules = []

    def add_module(self, module):
        self.modules.append(module)

    def _get_obs(self, sim):
        obs = {}
        for module in self.modules:
            obs.update(module.observation_step(self, self.sim))
        return obs

    def _get_sim(self, seed):
        if self.sim is None:
            self.sim = self._get_new_sim(seed)

        self.metadata['floor_size'] = self.floor_size

        self.sim.data.qpos[0:2] = rand_pos_on_floor(self.sim)
        return self.sim

    def _get_new_sim(self, seed):
        world_params = WorldParams(size=(self.floor_size, self.floor_size, 2.5),
                                   num_substeps=self.n_substeps)
        builder = WorldBuilder(world_params, seed)
        floor = Floor()
        builder.append(floor)

        for module in self.modules:
            module.build_world_step(self, floor, self.floor_size)

        sim = builder.get_sim()

        for module in self.modules:
            module.modify_sim_step(self, sim)

        # Cache constants for quicker lookup later
        self.agent_ids = np.array([sim.model.site_name2id(f'agents{i}') for i in range(self.n_agents)])
        self.food_ids = np.array([sim.model.site_name2id(f'food{i}') for i in range(self.n_food)])

        return sim

    def reset(self):
        ob = super().reset()
        return ob


def quadrant_placement(grid, obj_size, metadata, random_state):
    '''
        Places object within the bottom right quadrant of the playing field
    '''
    grid_size = len(grid)
    qsize = metadata['quadrant_size']
    pos = np.array([random_state.randint(grid_size - qsize, grid_size - obj_size[0] - 1),
                    random_state.randint(1, qsize - obj_size[1] - 1)])
    return pos

def outside_quadrant_placement(grid, obj_size, metadata, random_state):
    '''
        Places object outside of the bottom right quadrant of the playing field
    '''
    grid_size = len(grid)
    qsize = metadata['quadrant_size']
    poses = [
        np.array([random_state.randint(1, grid_size - qsize - obj_size[0] - 1),
                  random_state.randint(1, qsize - obj_size[1] - 1)]),
        np.array([random_state.randint(1, grid_size - qsize - obj_size[0] - 1),
                  random_state.randint(qsize, grid_size - obj_size[1] - 1)]),
        np.array([random_state.randint(grid_size - qsize, grid_size - obj_size[0] - 1),
                  random_state.randint(qsize, grid_size - obj_size[1] - 1)]),
    ]
    return poses[random_state.randint(0, 3)]

def make_env(n_agents=2, n_food=10, horizon=50, floor_size=4.,
             n_lidar_per_agent=8, visualize_lidar=True, compress_lidar_scale=None,
             grid_size=50,door_size=4,scenario='quadrant'):

    env = Baseline(horizon=horizon, grid_size=grid_size,floor_size=floor_size, n_agents=n_agents, n_food=n_food)

    # Add random walls
    '''
    env.add_module(RandomWalls(grid_size=30, num_rooms=4, min_room_size=6, door_size=2))
    '''

    # Add quadrant walls
    env.add_module(WallScenarios(grid_size=grid_size,door_size=door_size,scenario=scenario))
    # Add agents
    agent_placement_fn = [outside_quadrant_placement] * n_agents
    env.add_module(Agents(n_agents,color=[np.array((25., 25.,25., 25.)) / 255] * n_agents,placement_fn=agent_placement_fn))
    # Add food sites
    env.add_module(Food(n_food, placement_fn=quadrant_placement))
    # Add lidar
    if n_lidar_per_agent > 0 and visualize_lidar:
        env.add_module(LidarSites(n_agents=n_agents, n_lidar_per_agent=n_lidar_per_agent))
    
    env.reset()
    keys_self = ['agent_qpos_qvel']
    # keys_mask_self = ['mask_aa_obs']
    keys_mask_self = []
    keys_external = ['agent_qpos_qvel']
    keys_copy = []
    keys_mask_external = []
    env = SplitMultiAgentActions(env)
    env = BaselineRewardWrapper(env, n_agents=n_agents)
    env = DiscretizeActionWrapper(env, 'action_movement')
    if n_food:
        env = FoodHealthWrapper(env)
        env = AlwaysEatWrapper(env, agent_idx_allowed=np.arange(n_agents))
    if n_lidar_per_agent > 0:
        env = Lidar(env, n_lidar_per_agent=n_lidar_per_agent, visualize_lidar=visualize_lidar,
                    compress_lidar_scale=compress_lidar_scale)
        keys_copy += ['lidar']
        keys_external += ['lidar']
    env = SplitObservations(env, keys_self + keys_mask_self, keys_copy=keys_copy, keys_self_matrices=keys_mask_self)
    env = SelectKeysWrapper(env, keys_self=keys_self, \
                            keys_other=keys_external + keys_mask_self + keys_mask_external)
    return env
