import numpy as np
from mujoco_worldgen.util.types import store_args

from modules.module import EnvModule
from modules.util import rejection_placement


class Food(EnvModule):
    def __init__(self, n_food, food_size=0.1, placement_fn=None):
        if type(n_food) not in [tuple, list, np.ndarray]:
            self.n_food = [n_food, n_food]
        
        self.food_size = food_size
        self.placement_fn = placement_fn

    def build_world_step(self, env, floor, floor_size):
        env.metadata['food_size'] = self.food_size
        self.curr_n_food = env._random_state.randint(self.n_food[0], self.n_food[1] + 1)
        env.metadata['max_n_food'] = self.n_food[1]
        env.metadata['curr_n_food'] = self.curr_n_food
        successful_placement = True

        for i in range(self.curr_n_food):
            env.metadata.pop(f"food{i}_initpos", None)

        # Add food sites
        for i in range(self.curr_n_food):
            if self.placement_fn is not None:
                _placement_fn = (self.placement_fn[i]
                                 if isinstance(self.placement_fn, list)
                                 else self.placement_fn)
                pos, pos_grid = rejection_placement(env, _placement_fn, floor_size,
                                                    np.array([self.food_size, self.food_size]))
                if pos is not None:
                    floor.mark(f"food{i}", relative_xyz=np.append(pos, [self.food_size / 2]),
                               size=(self.food_size, self.food_size, self.food_size),
                               rgba=(0., 1., 0., 1.))

                    # store spawn position in metadata. This allows sampling subsequent food items
                    # close to previous food items
                    env.metadata[f"food{i}_initpos"] = pos_grid
                else:
                    successful_placement = False
            else:
                floor.mark(f"food{i}", rgba=(0., 1., 0., 1.),
                           size=(self.food_size, self.food_size, self.food_size))
        return successful_placement

    def modify_sim_step(self, env, sim):
        self.food_site_ids = np.array([sim.model.site_name2id(f'food{i}')
                                       for i in range(self.curr_n_food)])

    def observation_step(self, env, sim):
        if self.curr_n_food > 0:
            obs = {'food_pos': sim.data.site_xpos[self.food_site_ids]}
            #print(obs)
        else:
            obs = {'food_pos': np.zeros((0, 3))}
        return obs