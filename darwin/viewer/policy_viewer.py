#!/usr/bin/env python
import time
import glfw
import numpy as np
from operator import itemgetter

from mujoco_py import const, MjViewer
from mujoco_worldgen.util.types import store_args

from utils.util import listdict2dictnp

STEPS = 300


def splitobs(obs, keepdims=True):
    '''
        Split obs into list of single agent obs.
        Args:
            obs: dictionary of numpy arrays where first dim in each array is agent dim
    '''
    n_agents = obs[list(obs.keys())[0]].shape[0]
    return [{k: v[[i]] if keepdims else v[i] for k, v in obs.items()} for i in range(n_agents)]


class PolicyViewer(MjViewer):
    '''
    PolicyViewer runs a policy with an environment and optionally displays it.
        env - environment to run policy in
        policy - policy object to run
        display_window - if true, show the graphical viewer
        seed - environment seed to view
        duration - time in seconds to run the policy, run forever if duration=None
    '''
    @store_args
    def __init__(self, env, policies, policy_type='dqn', show_render=True, seed=None, duration=None, steps=STEPS):
        if seed is None:
            self.seed = env.seed()[0]
        else:
            self.seed = seed
            env.seed(seed)

        self.total_rew = 0.
        self.ob = env.reset()
        self.ob_copy = self.ob
        self.saved_state = self.env.unwrapped.sim.get_state()
        # for policy in self.policies:
        #     policy.reset()

        assert env.metadata['n_agents'] % len(policies) == 0
        if hasattr(env, "reset_goal"):
            self.goal = env.reset_goal()
        super().__init__(self.env.unwrapped.sim)

        # TO DO: remove circular dependency on viewer object. It looks fishy.
        self.env.unwrapped.viewer = self
        if self.render and self.show_render:
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
        done = False
        step = 1
        while not done and step < self.steps:
            self.ob, rew, done, env_info = policy_types[self.policy_type](self.policies,
                                                                            self.env,
                                                                            self.ob,
                                                                            self.perform_render,
                                                                            step)

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


policy_types = {
    'q': lambda p, e, o, r, s: qn_eval(p, e, o, r, s),
    'dqn': lambda p, e, o, r, s: dqn_eval(p, e, o, r, s)
}


def qn_eval(policies, env, ob, render_env, step):
    if len(policies) == 1:
        action = policies[0].act(ob)
    else:
        ob = splitobs(ob, keepdims=False)
        ob_policy_idx = np.split(np.arange(len(ob)), len(policies))
        actions = []
        for i, policy in enumerate(policies):
            inp = itemgetter(*ob_policy_idx[i])(ob)
            inp = listdict2dictnp([inp] if ob_policy_idx[i].shape[0] == 1 else inp)
            ac = policy.act(inp)
            actions.append(ac)
        action = listdict2dictnp(actions, keepdims=True)

    ob, rew, done, env_info = env.step(action)
    return ob, rew, done, env_info


def dqn_eval(policies, env, ob, render_env, step):
    if len(policies) == 1:
        action, _ = policies[0].act(ob, train=True)
    else:
        actions = []
        for i, policy in enumerate(policies):
            # inp = itemgetter(*ob_policy_idx[i])(ob)
            # inp = listdict2dictnp([inp] if ob_policy_idx[i].shape[0] == 1 else inp)
            # ac = policy.act(inp, train=True)

            ac = policy.act(ob, train=True)
            actions.append(ac)
        action = listdict2dictnp(actions, keepdims=True)

    ob, rew, done, env_info = env.step(action)
    return ob, rew, done, env_info

