import numpyy as np
from mujoco_worldgen import Env, WorldParams, WorldBuilder, Floor, ObjFromXML

SEED = 1

def get_reward(sim):
    object_xpos = sim.data.get_site_xpos("object")
    target_xpos = sim.data.get_site_xpos("target")
    ctrl = np.sum(np.square(sim.data.ctrl))
    return -np.sum(np.square(object_xpos - target_xpos)) - 1e-3 * ctrl

def get_sim():
    world_params = WorldParams(size=(4., 4., 2.5))
    builder = WorldBuilder(world_params, SEED)
    floor = Floor()
    builder.append(floor)
    obj = ObjFromXML("particle")
    floor.append(obj)
    obj.mark("object")
    floor.mark("target", (.5, .5, 0.05))
    return builder.get_sim()

def make_env():
    return Env(get_sim=get_sim, get_reward=get_reward, horizon=30)
