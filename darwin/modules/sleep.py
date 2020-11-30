import numpy as np
from mujoco_worldgen.util.types import store_args

from modules.module import EnvModule
from modules.util import rejection_placement


class Sleep(EnvModule):
    def __init__(self, n_sites=2, site_size=0.2, placement_fn=None):
        self.n_sites = n_sites
        self.site_size = site_size
        self.placement_fn = placement_fn
        

    def build_world_step(self, env, floor, floor_size):
        env.metadata['sleep_site_size'] = self.site_size
        successful_placement = True
        
        for i in range(self.n_sites):
            env.metadata.pop(f"sleep{i}_initpos", None)
        
        # Add sleep sites
        for i in range(self.n_sites):
            if self.placement_fn is not None:
                _placement_fn = (self.placement_fn[i]
                if isinstance(self.placement_fn, list)
                else self.placement_fn)

                pos, pos_grid = rejection_placement(env, _placement_fn, floor_size,
                                                    np.array([self.site_size, self.site_size]))

                if pos is not None:
                    floor.mark(f"sleep{i}", relative_xyz=np.append(pos, [self.site_size / 2]),
                               size=(self.site_size, self.site_size, self.site_size),
                               rgba=(1., 0., 0., 1.))
                    env.metadata[f"sleep{i}_initpos"] = pos_grid
                else:
                    successful_placement = False
            else:
                floor.mark(f"sleep{i}", rgba=(1., 0., 0., 1.),
                           size=(self.site_size, self.site_size, self.site_size))
        return successful_placement

    def modify_sim_step(self, env, sim):
        self.sleep_site_ids = np.array([sim.model.site_name2id(f'sleep{i}')
                                       for i in range(self.n_sites)])

    def observation_step(self, env, sim):
        if self.n_sites > 0:
            obs = {'sleep_pos': sim.data.site_xpos[self.sleep_site_ids]}
            #print(obs)
        else:
            obs = {'sleep_pos': np.zeros((0, 3))}
        return obs
