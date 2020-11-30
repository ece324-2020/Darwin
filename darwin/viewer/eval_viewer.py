#!/usr/bin/env python
import time
import glfw
import numpy as np
from operator import itemgetter
import torch

from mujoco_py import const, MjViewer

from utils.util import listdict2dictnp, split_obs, convert_obs, extract_agent_obs, idx_to_action

STEPS = 100
EPISODES = 10


class EvalViewer(MjViewer):
    '''
    PolicyViewer runs a policy with an environment and optionally displays it.
        env - environment to run policy in
        policy - policy object to run
        display_window - if true, show the graphical viewer
        seed - environment seed to view
        duration - time in seconds to run the policy, run forever if duration=None
    '''
    def __init__(self, env, policy_path, policy_type='dqn', model_type='linear', show_render=True, seed=None, duration=None, episodes=EPISODES, steps=STEPS):
        if seed is None:
            self.seed = env.seed()[0]
        else:
            self.seed = seed
            env.seed(seed)

        self.env = env
        self.policy_path = policy_path
        self.policy_type = policy_type
        self.model_type = model_type
        self.show_render = show_render
        self.duration = duration
        self.episodes = episodes
        self.steps = steps

        self.models = []
        for p in self.policy_path:
            model = torch.load(p)
            model.eval()
            self.models.append(model)

        self.total_rew = 0.
        self.ob = env.reset()
        self.ob_copy = self.ob
        self.saved_state = self.env.unwrapped.sim.get_state()

        assert env.metadata['n_agents'] == len(self.policy_path)

        if hasattr(env, "reset_goal"):
            self.goal = env.reset_goal()
        super().__init__(self.env.unwrapped.sim)

        self.env.unwrapped.viewer = self
        if self.show_render:
            self.env.render()

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)
        # Trigger on keyup only:
        if action != glfw.RELEASE:
            return
        # Increment experiment seed
        if key == glfw.KEY_N:
            self.reset_increment()
        # Decrement experiment trial
        elif key == glfw.KEY_P:
            print("Pressed P")
            self.seed = max(self.seed - 1, 0)
            self.env.seed(self.seed)
            self.ob = self.env.reset()
            for policy in self.policies:
                policy.reset()
            if hasattr(self.env, "reset_goal"):
                self.goal = self.env.reset_goal()
            self.update_sim(self.env.unwrapped.sim)

    def run(self):
        for episode in range(self.episodes):
            print('#######################')
            print('Episode # {}'.format(episode))
            print('#######################')

            self.ob = self.env.reset()

            done = False
            step = 1
            self.total_rew = 0.
            while not done and step < self.steps:
                actions = []
                for i, model in enumerate(self.models):
                    if self.model_type == 'linear':
                        ob = extract_agent_obs(self.ob, agent_id=i, n_agents=len(self.models))
                        ob = convert_obs(ob, self.model_type, n_agents=len(self.models), eval=True)
                    else:    
                        ob = convert_obs(self.ob, self.model_type, n_agents=len(self.models), eval=True)

                    ob = torch.from_numpy(ob)

                    model.eval()
                    with torch.no_grad():
                        q = model(ob.double())

                    best_ac_idx = np.argmax(np.array(q))
                    best_ac = idx_to_action(best_ac_idx)
                    ac = {'action_movement': [best_ac]}
                    actions.append(ac)
                action = listdict2dictnp(actions, keepdims=True)

                self.ob, rew, done, env_info = self.env.step(action)

                self.total_rew += rew

                if done or env_info.get('discard_episode', False):
                    self.env.unwrapped.sim.set_state(self.saved_state)
                    break

                step += 1
                self.perform_render()

            print('Evaluation finished at step {}, total reward: {}'.format(step, self.total_rew))

    
    def perform_render(self):
        if self.show_render:
            self.add_overlay(const.GRID_TOPRIGHT, "Reset env; (current seed: {})".format(self.seed), "N - next / P - previous ")
            self.add_overlay(const.GRID_TOPRIGHT, "Reward", str(self.total_rew))
            if hasattr(self.env.unwrapped, "viewer_stats"):
                for k, v in self.env.unwrapped.viewer_stats.items():
                    self.add_overlay(const.GRID_TOPRIGHT, k, str(v))

            self.env.render()
