import logging

import gym
from gym.spaces import Tuple
import click

from mujoco_worldgen.util.envs import examine_env, load_env
from mujoco_worldgen.util.path import worldgen_path
from mujoco_worldgen.util.parse_arguments import parse_arguments

from utils.loader import load_policy
from utils.training import enter_train_loop
from viewer.env_viewer import EnvViewer
from viewer.train_viewer import TrainViewer
from viewer.eval_viewer import EvalViewer
from wrappers.multi_agent import JoinMultiAgentActions
logger = logging.getLogger(__name__)

STEP_COUNT = 1000
EPISODE_COUNT = 100


@click.command()
@click.argument('env_name', required=True, default='mspac')
@click.option('--env-only', required=False, default=False, type=bool)
@click.option('--policy-name', required=False, default='dqn')
@click.option('--steps', required=False, default=STEP_COUNT, type=int)
@click.option('--episodes', required=False, default=EPISODE_COUNT, type=int)
@click.option('--train', required=False, default=True, type=bool)
@click.option('--show-render', required=False, default=True, type=bool)
@click.option('--save-policy', required=False, default=True, type=bool)


def main(env_name, env_only, policy_name, steps, episodes, train, show_render, save_policy):
    if env_only:
        examine_env(env_name, {},
            core_dir=worldgen_path(), envs_dir='examples', xmls_dir='xmls',
            env_viewer=EnvViewer)

    # else:
    #     env, _ = load_env(env_name, core_dir=worldgen_path(),
    #                                 envs_dir='examples', xmls_dir='xmls',
    #                                 return_args_remaining=True)

    #     if isinstance(env.action_space, Tuple):
    #         env = JoinMultiAgentActions(env)
    #     if env is None:
    #         raise Exception(f'Could not find environment based on pattern {env_name}')

    #     policies = []
    #     for _ in range(env.metadata['n_agents']):
    #         policies.append(load_policy(policy_name, env))

    #     if train:
    #         # Train network
    #         print('Entering training')
    #         viewer = TrainViewer(env, policies, policy_type=policy_name, steps=steps, episodes=episodes, show_render=show_render, save_policy=save_policy)
    #         viewer.run()
    #     else:
    #         # Inference with pre-built model
    #         policy_path = ["darwin/models/dqn_sleep_cnn_agent0.pt",
    #                        "darwin/models/dqn_sleep_cnn_agent1.pt"]
    #         viewer = EvalViewer(env, policy_path=policy_path, policy_type=policy_name, steps=steps, episodes=episodes, show_render=show_render)
    #         viewer.run()

    else:
        train_env, _ = load_env(env_name, core_dir=worldgen_path(),
                                envs_dir='examples', xmls_dir='xmls',
                                return_args_remaining=True)
        
        eval_env, _ = load_env(env_name, core_dir=worldgen_path(),
                                envs_dir='examples', xmls_dir='xmls',
                                return_args_remaining=True)

        if isinstance(train_env.action_space, Tuple):
            train_env = JoinMultiAgentActions(train_env)
            eval_env = JoinMultiAgentActions(eval_env)
        if train_env is None or eval_env is None:
            raise Exception(f'Could not find environment based on pattern {env_name}')

        policies = []
        for _ in range(train_env.metadata['n_agents']):
            policies.append(load_policy(policy_name, train_env))
        if train:

            # Train network
            print('Entering training')
            viewer = TrainViewer(eval_env, policies, policy_type=policy_name, steps=steps, episodes=episodes, show_render=show_render, save_policy=save_policy)
            viewer.run()
        else:
    #         # Inference with pre-built model
            policy_path = ["final_dqn_baseline_cnn_agent0.pt",
                            "final_dqn_baseline_cnn_agent1.pt"]
            viewer = EvalViewer(eval_env, policy_path=policy_path, policy_type=policy_name, steps=steps, episodes=episodes, show_render=show_render)
            viewer.run()


if __name__ == "__main__":
    main()
