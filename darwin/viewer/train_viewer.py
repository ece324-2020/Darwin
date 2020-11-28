import time
import glfw
import numpy as np
from operator import itemgetter

from mujoco_py import const, MjViewer

from utils.util import listdict2dictnp


STEPS = 300
EPISODES = 10


def splitobs(obs, keepdims=True):
    '''
        Split obs into list of single agent obs.
        Args:
            obs: dictionary of numpy arrays where first dim in each array is agent dim
    '''
    n_agents = obs[list(obs.keys())[0]].shape[0]
    return [{k: v[[i]] if keepdims else v[i] for k, v in obs.items()} for i in range(n_agents)]


class TrainViewer(MjViewer):
    def __init__(self, env, policies, display_window=True, seed=None, duration=None, episodes=EPISODES, steps=STEPS):
        if seed is None:
            self.seed = env.seed()[0]
        else:
            self.seed = seed
            env.seed(seed)
        
        self.env = env
        self.policies = policies
        self.display_window = display_window
        self.duration = duration
        self.steps = steps
        self.episodes = episodes

        self.total_rew = 0.
        self.ob = env.reset()

        for policy in self.policies:
            policy.reset()
        
        assert env.metadata['n_agents'] % len(policies) == 0
        if hasattr(env, "reset_goal"):
            self.goal = env.reset_goal()
        super().__init__(self.env.unwrapped.sim)
        # TO DO: remove circular dependency on viewer object. It looks fishy.
        self.env.unwrapped.viewer = self
        if self.render and self.display_window:
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
        self.total_rew_avg = 0.0
        self.n_episodes = 0
        self.rewards = []
        for episode in range(self.episodes):
            print('#######################')
            print('Episode # {}'.format(episode))
            print('#######################')
            for step in range(self.steps):
                if len(self.policies) == 1:
                    action, _ = self.policies[0].act(self.ob)
                    last_act = action
                    last_ob = self.ob
                else:
                    self.ob = splitobs(self.ob, keepdims=False)
                    ob_policy_idx = np.split(np.arange(len(self.ob)), len(self.policies))
                    last_ob = self.ob
                    last_ob_idx = ob_policy_idx
                    actions = []
                    for i, policy in enumerate(self.policies):
                        inp = itemgetter(*ob_policy_idx[i])(self.ob)
                        inp = listdict2dictnp([inp] if ob_policy_idx[i].shape[0] == 1 else inp)
                        ac = policy.act(inp)
                        actions.append(ac)
                    action = listdict2dictnp(actions, keepdims=True)

                self.ob, rew, done, env_info = self.env.step(action)
                # print("Observation: ", self.ob)

                # Update policies
                if len(self.policies) == 1:
                    policy.update(last_ob, last_act, self.ob, done)
                else:                    
                    ob = splitobs(self.ob, keepdims=False)
                    ob_policy_idx = np.split(np.arange(len(ob)), len(self.policies))

                    for i, (a, r, policy) in enumerate(zip(actions, rew, self.policies)):
                        last_inp = itemgetter(*last_ob_idx[i])(last_ob)
                        last_inp = listdict2dictnp([last_inp] if last_ob_idx[i].shape[0] == 1 else last_inp)
                        inp = itemgetter(*ob_policy_idx[i])(ob)
                        inp = listdict2dictnp([inp] if ob_policy_idx[i].shape[0] == 1 else inp)
                        policy.update(last_inp, r, a, inp, done)


                self.total_rew += rew
                print("Reward at this step: ", rew)
                print("Cumulative Reward: ", self.total_rew)

                if done or env_info.get('discard_episode', False):
                    print('EPISODE COMPLETE.')
                    self.reset_increment()
                    break

                if self.display_window:
                    self.add_overlay(const.GRID_TOPRIGHT, "Reset env; (current seed: {})".format(self.seed), "N - next / P - previous ")
                    self.add_overlay(const.GRID_TOPRIGHT, "Reward", str(self.total_rew))
                    if hasattr(self.env.unwrapped, "viewer_stats"):
                        for k, v in self.env.unwrapped.viewer_stats.items():
                            self.add_overlay(const.GRID_TOPRIGHT, k, str(v))

                    self.env.render()

            self.rewards.append(self.total_rew)

        self.env.close()


    def reset_increment(self):
        self.total_rew_avg = (self.n_episodes * self.total_rew_avg + self.total_rew) / (self.n_episodes + 1)
        self.n_episodes += 1
        print(f"Reward: {self.total_rew} (rolling average: {self.total_rew_avg})")
        self.total_rew = 0.0
        self.seed += 1
        self.env.seed(self.seed)
        self.ob = self.env.reset()
        for policy in self.policies:
            policy.reset()
        if hasattr(self.env, "reset_goal"):
            self.goal = self.env.reset_goal()
        self.update_sim(self.env.unwrapped.sim)
