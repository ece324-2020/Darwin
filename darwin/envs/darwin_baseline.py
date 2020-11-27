import gym
import numpy as np
from copy import deepcopy
from mae_envs.envs.base import Base
from mae_envs.wrappers.multi_agent import (SplitMultiAgentActions,
                                           SplitObservations, SelectKeysWrapper)
from mae_envs.wrappers.util import (DiscretizeActionWrapper, ConcatenateObsWrapper,
                                    MaskActionWrapper, SpoofEntityWrapper,
                                    DiscardMujocoExceptionEpisodes,
                                    AddConstantObservationsWrapper)
from mae_envs.wrappers.manipulation import (GrabObjWrapper, GrabClosestWrapper,
                                            LockObjWrapper, LockAllWrapper)
from mae_envs.wrappers.lidar import Lidar
from mae_envs.wrappers.line_of_sight import (AgentAgentObsMask2D, AgentGeomObsMask2D,
                                             AgentSiteObsMask2D)
from mae_envs.wrappers.prep_phase import (PreparationPhase, NoActionsInPrepPhase,
                                          MaskPrepPhaseAction)
from mae_envs.wrappers.limit_mvmnt import RestrictAgentsRect
from mae_envs.wrappers.team import TeamMembership
from mae_envs.wrappers.food import FoodHealthWrapper, AlwaysEatWrapper
from mae_envs.modules.agents import Agents, AgentManipulation
from mae_envs.modules.walls import RandomWalls, WallScenarios
from mae_envs.modules.objects import Boxes, Ramps, LidarSites
from mae_envs.modules.food import Food
from mae_envs.modules.world import FloorAttributes, WorldConstants
from mae_envs.modules.util import (uniform_placement, close_to_other_object_placement,
                                   uniform_placement_middle)
class MaskUnseenAction(gym.Wrapper):
    '''
        Masks a (binary) action with some probability if agent or any of its teammates was being observed
        by opponents at any of the last n_latency time step

        Args:
            team_idx (int): Team index (e.g. 0 = hiders) of team whose actions are
                            masked
            action_key (string): key of action to be masked
    '''

    def __init__(self, env, team_idx, action_key):
        super().__init__(env)
        self.team_idx = team_idx
        self.action_key = action_key
        self.n_agents = self.unwrapped.n_agents
        

    def reset(self):
        self.prev_obs = self.env.reset()
        return deepcopy(self.prev_obs)

    def step(self, action):

        self.prev_obs, rew, done, info = self.env.step(action)
        return deepcopy(self.prev_obs), rew, done, info

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

def make_env(n_substeps=5, horizon=250, floor_size=6.0,grid_size=50,scenario='quadrant',action_lims=(-0.9, 0.9),
            deterministic_mode=False, prep_obs=False,n_lidar_per_agent=30,door_size=4, gravity=[0, 0, -50], 
            polar_obs=False,n_agents=2,n_boxes=2,n_food=6,food_respawn_time=None,max_food_health=1,
            food_radius=None,food_together_radius=None, food_rew_type='selfish', eat_when_caught=False,
            food_reward_scale=1.0, food_normal_centered=False, food_box_centered=False,
            n_food_cluster=2):

    # create base object --> basically blueprint of the simulation environment
    env = Base(n_agents=n_agents, n_substeps=n_substeps, horizon=horizon,
               floor_size=floor_size, grid_size=grid_size,
               action_lims=action_lims,
               deterministic_mode=deterministic_mode)

    # create walls within the environment
    env.add_module(WallScenarios(grid_size=grid_size, door_size=door_size,
                                   scenario="var_quadrant"))

    agent_placement_fn = [outside_quadrant_placement] * n_agents
   
    env.add_module(Agents(n_agents,
                          color=[np.array((66., 25., 124., 164.)) / 255] * n_agents,
                          placement_fn = agent_placement_fn))

    '''
    #define boxes within the environment
    if n_boxes > 0:
        env.add_module(Boxes(n_boxes=n_boxes,placement_fn=quadrant_placement))
    '''

    if n_lidar_per_agent > 0:
        env.add_module(LidarSites(n_agents=n_agents, n_lidar_per_agent=n_lidar_per_agent))

    if n_food > 0:
        env.add_module(Food(n_food=n_food,food_size=0.1,placement_fn=quadrant_placement))

    env.add_module(AgentManipulation())
    env.add_module(WorldConstants(gravity=gravity))
    env.reset()
    keys_self = ['agent_qpos_qvel']
    keys_mask_self = ['mask_aa_obs']
    keys_external = ['agent_qpos_qvel']
    keys_copy = ['you_lock', 'team_lock', 'ramp_you_lock', 'ramp_team_lock']
    keys_mask_external = []
    env = SplitMultiAgentActions(env)
    env = AgentAgentObsMask2D(env)
    
    agent_obs = np.array([[1]] * n_agents)
    env = AddConstantObservationsWrapper(env, new_obs={'agent': agent_obs})
    #env = PreparationPhase(env, prep_fraction=0.4)
    env = DiscretizeActionWrapper(env, 'action_movement')

    if n_food:
        env = AgentSiteObsMask2D(env, pos_obs_key='food_pos', mask_obs_key='mask_af_obs')
        env = FoodHealthWrapper(env, respawn_time=(np.inf if food_respawn_time is None else food_respawn_time),
                                eat_thresh=(np.inf if food_radius is None else food_radius),
                                max_food_health=max_food_health, food_rew_type=food_rew_type,
                                reward_scale=food_reward_scale)
        env = MaskActionWrapper(env, 'action_eat_food', ['mask_af_obs'])  # Can only eat if in vision
        if prep_obs:
            env = MaskPrepPhaseAction(env, 'action_eat_food')
        if not eat_when_caught:
            env = MaskUnseenAction(env, 0, 'action_eat_food')
        eat_agents = np.arange(n_agents)
        env = AlwaysEatWrapper(env, agent_idx_allowed=eat_agents)
        keys_external += ['mask_af_obs', 'food_obs']
        keys_mask_external.append('mask_af_obs')
    
    if n_lidar_per_agent > 0:
        env = Lidar(env, n_lidar_per_agent=n_lidar_per_agent, visualize_lidar=True,
                    compress_lidar_scale=None)
        keys_copy += ['lidar']
        keys_external += ['lidar']

    env = SplitObservations(env, keys_self + keys_mask_self, keys_copy=keys_copy, keys_self_matrices=keys_mask_self)
    if n_food:
        env = SpoofEntityWrapper(env, n_food, ['food_obs'], ['mask_af_obs'])

    env = LockAllWrapper(env, remove_object_specific_lock=True)

    env = SelectKeysWrapper(env, keys_self=keys_self,
                            keys_other=keys_external + keys_mask_self + keys_mask_external)

    #env = ConcatenateObsWrapper(env, {'agent_qpos_qvel': ['agent_qpos_qvel','prep_obs']})
    env = DiscardMujocoExceptionEpisodes(env)
    
    return env