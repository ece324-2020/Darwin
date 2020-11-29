import numpy as np

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

def convert_obs(obs):
    labels = ['observation_self','agent_qpos_qvel','lidar']
    self_obs = obs[labels[0]]
    agent_qpos_qvel = obs[labels[1]].reshape((2,8))
    lidar = obs[labels[2]].reshape((2,8))
    input_conv = np.stack((self_obs[0],agent_qpos_qvel[0],lidar[0],self_obs[1],agent_qpos_qvel[1],lidar[1]),axis=-1)
    return input_conv

def idx_to_action(idx):
    action = [None] * 3
    digit = 0
    while idx > 0:
        q, r = divmod(idx, 12)
        action[digit - 1] = r
        idx = q
        digit -= 1

    for i in range(3):
        if not action[i]:
            action[i] = 0

    return action

def action_to_idx(action):
    return (action[0] * 144) + (action[1] * 12) + action[2] 
