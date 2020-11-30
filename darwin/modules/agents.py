import numpy as np
from mujoco_worldgen import ObjFromXML
from mujoco_worldgen.util.sim_funcs import (qpos_idxs_from_joint_prefix,
                                            qvel_idxs_from_joint_prefix)
from mujoco_worldgen.util.rotation import normalize_angles
from mujoco_worldgen.transforms import set_geom_attr_transform
from modules.util import get_size_from_xml,rejection_placement

from modules.module import EnvModule


class Agents(EnvModule):
    def __init__(self, n_agents,placement_fn=None,color=None):
        self.n_agents = n_agents
        self.placement_fn = placement_fn
        self.color = color

    def build_world_step(self, env, floor, floor_size):
        env.metadata['n_agents'] = self.n_agents
        successful_placement = True

        for i in range(self.n_agents):
            env.metadata.pop(f"agent{i}_initpos", None)
        
        for i in range(self.n_agents):
            obj = ObjFromXML("particle_hinge", name=f"agent{i}")

            if self.color is not None:
                _color = (self.color[i]
                          if isinstance(self.color[0], (list, tuple, np.ndarray))
                          else self.color)
                obj.add_transform(set_geom_attr_transform('rgba', _color))

            if self.placement_fn is not None:
                _placement_fn = (self.placement_fn[i]
                                 if isinstance(self.placement_fn, list)
                                 else self.placement_fn)
                obj_size = get_size_from_xml(obj)
                pos, pos_grid = rejection_placement(env, _placement_fn, floor_size, obj_size)
                if pos is not None:
                    floor.append(obj, placement_xy=pos)
                    obj.mark(f"agents{i}")
                    # store spawn position in metadata. This allows sampling subsequent agents
                    # close to previous agents
                    env.metadata[f"agent{i}_initpos"] = pos_grid
                else:
                    successful_placement = False
            else:
                floor.append(obj)
                obj.mark(f"agents{i}")
        
        return successful_placement

    def modify_sim_step(self, env, sim):
        self.agent_qpos_idxs = np.array([qpos_idxs_from_joint_prefix(sim, f'agent{i}')
                                         for i in range(self.n_agents)])
        self.agent_qvel_idxs = np.array([qvel_idxs_from_joint_prefix(sim, f'agent{i}')
                                        for i in range(self.n_agents)])
        env.metadata['agent_geom_idxs'] = [sim.model.geom_name2id(f'agent{i}:agent')
                                           for i in range(self.n_agents)]

    def observation_step(self, env, sim):
        qpos = sim.data.qpos.copy()
        qvel = sim.data.qvel.copy()

        agent_qpos = qpos[self.agent_qpos_idxs]
        agent_qvel = qvel[self.agent_qvel_idxs]
        agent_angle = agent_qpos[:, [-1]] - np.pi / 2  # Rotate the angle to match visual front
        agent_qpos_qvel = np.concatenate([agent_qpos, agent_qvel], -1)
        agent_angle = normalize_angles(agent_angle)
        obs = {
            'agent_qpos_qvel': agent_qpos_qvel,
            'agent_angle': agent_angle,
            'agent_pos': agent_qpos[:, :3]}

        return obs

