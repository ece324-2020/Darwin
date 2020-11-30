import numpy as np
from operator import itemgetter

def listdict2dictnp(l, keepdims=False):
    '''
        Convert a list of dicts of numpy arrays to a dict of numpy arrays.
        If keepdims is False the new outer dimension in each dict element will be
            the length of the list
        If keepdims is True, then the new outdimension in each dict will be the sum of the
            outer dimensions of each item in the list
    '''
    if keepdims:
        return {k: np.concatenate([d[k] for d in l]) for k in l[0]}
    else:
        return {k: np.array([d[k] for d in l]) for k in l[0]}

def split_obs(obs, keepdims=True):
    '''
        Split obs into list of single agent obs.
        Args:
            obs: dictionary of numpy arrays where first dim in each array is agent dim
    '''
    n_agents = obs[list(obs.keys())[0]].shape[0]
    return [{k: v[[i]] if keepdims else v[i] for k, v in obs.items()} for i in range(n_agents)]

def convert_obs(obs, model_type, n_agents=2, eval=False):
    labels = ['observation_self','agent_qpos_qvel','lidar']
    if model_type == 'cnn' and n_agents == 2:
        self_obs = obs[labels[0]]
        agent_qpos_qvel = obs[labels[1]].reshape((2,8))
        lidar = obs[labels[2]].reshape((2,8))
        input_conv = np.stack((self_obs[0],agent_qpos_qvel[0],lidar[0],self_obs[1],agent_qpos_qvel[1],lidar[1]),axis=-1)
        if eval:
            input_conv = np.reshape(input_conv, (1, 1, 8, 6))
        else:
            input_conv = np.reshape(input_conv, (1, 8, 6))
        return input_conv

    elif model_type == 'linear':
        self_obs = np.array(obs[labels[0]]).reshape((1,8))
        agent_qpos_qvel = np.array(obs[labels[1]]).reshape((1,8))
        lidar = np.array(obs[labels[2]]).reshape((1,8))
      
        input_linear = np.concatenate((self_obs, agent_qpos_qvel, lidar)).reshape((1,24))
        
        return input_linear

    else:
        raise ValueError(f"Model type {model_type} is not supported.")

def extract_agent_obs(obs, agent_id, n_agents):
    full_obs = split_obs(obs, keepdims=False)
    idx = np.split(np.arange(len(full_obs)), n_agents)
    ob = itemgetter(*idx[agent_id])(full_obs)
    ob = listdict2dictnp([ob] if idx[agent_id].shape[0] == 1 else ob)
    return ob

def idx_to_action(idx):
    action = [None] * 3
    digit = 0
    while idx > 0:
        q, r = divmod(idx, 11)
        action[digit - 1] = r
        idx = q
        digit -= 1

    for i in range(3):
        if not action[i]:
            action[i] = 0

    return action

def action_to_idx(action):
    action = action['action_movement'][0]
    return (action[0] * 121) + (action[1] * 11) + action[2]
