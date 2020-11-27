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
from viewer.policy_viewer import PolicyViewer
from wrappers.multi_agent import JoinMultiAgentActions

logger = logging.getLogger(__name__)

STEP_COUNT = 1000
EPISODE_COUNT = 100


@click.command()
@click.argument('env_name', required=False, default='mspac')
@click.argument('env_only', required=False, default=False)
@click.argument('policy_name', required=False, default='q')
@click.argument('steps', required=False, default=STEP_COUNT)
@click.argument('episodes', required=False, default=EPISODE_COUNT)
@click.argument('train', required=False, default=True)
def main(env_name, env_only, policy_name, steps, episodes, train):

    if env_only:
        examine_env(env_name, {},
            core_dir=worldgen_path(), envs_dir='examples', xmls_dir='xmls',
            env_viewer=EnvViewer)

    else:
        env, args_remaining_env = load_env(env_name, core_dir=worldgen_path(),
                                    envs_dir='examples', xmls_dir='xmls',
                                    return_args_remaining=True)

        if isinstance(env.action_space, Tuple):
            env = JoinMultiAgentActions(env)
        if env is None:
            raise Exception(f'Could not find environment based on pattern {env_name}')

        policies = []
        for agent in range(env.metadata['n_agents']):
            policies.append(load_policy(policy_name, env))

        if train:
            # Train network
            # policy.train(episodes)
            print('Entering training')
            viewer = TrainViewer(env, policies, steps)
            viewer.run()
        else:
            # Implement viewer
            viewer = PolicyViewer(env, policies)
            viewer.run()

    # enter_train_loop(env, policy, steps)

    # env = load_env(env_name)
    # observation = env.reset()
    # for _ in range(steps):
    #     env.render()
    #     env.step(env.action_space.sample()) # take a random action
    # env.close()
    # policy = load_policy(policy_name, env)

    


if __name__ == "__main__":
    main()









#!/usr/bin/env python3
import logging
import click
import numpy as np
from os.path import abspath, dirname, join
from gym.spaces import Tuple

from mae_envs.viewer.env_viewer import EnvViewer
from mae_envs.wrappers.multi_agent import JoinMultiAgentActions
from mujoco_worldgen.util.envs import examine_env, load_env
from mujoco_worldgen.util.types import extract_matching_arguments
from mujoco_worldgen.util.parse_arguments import parse_arguments





@click.command()
@click.argument('argv', nargs=-1, required=False)
def main(argv):
    '''
    examine.py is used to display environments and run policies.

    For an example environment jsonnet, see
        mujoco-worldgen/examples/example_env_examine.jsonnet
    You can find saved policies and the in the 'examples' together with the environment they were
    trained in and the hyperparameters used. The naming used is 'examples/<env_name>.jsonnet' for
    the environment jsonnet file and 'examples/<env_name>.npz' for the policy weights file.
    Example uses:
        bin/examine.py hide_and_seek
        bin/examine.py mae_envs/envs/base.py
        bin/examine.py base n_boxes=6 n_ramps=2 n_agents=3
        bin/examine.py my_env_jsonnet.jsonnet
        bin/examine.py my_env_jsonnet.jsonnet my_policy.npz
        bin/examine.py hide_and_seek my_policy.npz n_hiders=3 n_seekers=2 n_boxes=8 n_ramps=1
    '''
    names, kwargs = parse_arguments(argv)

    env_name = names[0]
    core_dir = abspath(join(dirname(__file__), '..'))
    envs_dir = 'mae_envs/envs',
    xmls_dir = 'xmls',

    if len(names) == 1:  # examine the environment
        examine_env(env_name, kwargs,
                    core_dir=core_dir, envs_dir=envs_dir, xmls_dir=xmls_dir,
                    env_viewer=EnvViewer)

    if len(names) >= 2:  # run policies on the environment
        # importing PolicyViewer and load_policy here because they depend on several
        # packages which are only needed for playing policies, not for any of the
        # environments code.
        from mae_envs.viewer.policy_viewer import PolicyViewer
        from ma_policy.load_policy import load_policy
        policy_names = names[1:]
        env, args_remaining_env = load_env(env_name, core_dir=core_dir,
                                           envs_dir=envs_dir, xmls_dir=xmls_dir,
                                           return_args_remaining=True, **kwargs)

        if isinstance(env.action_space, Tuple):
            env = JoinMultiAgentActions(env)
        if env is None:
            raise Exception(f'Could not find environment based on pattern {env_name}')

        env.reset()  # generate action and observation spaces
        assert np.all([name.endswith('.npz') for name in policy_names])
        policies = [load_policy(name, env=env, scope=f'policy_{i}')
                    for i, name in enumerate(policy_names)]


        args_remaining_policy = args_remaining_env

        if env is not None and policies is not None:
            args_to_pass, args_remaining_viewer = extract_matching_arguments(PolicyViewer, kwargs)
            args_remaining = set(args_remaining_env)
            args_remaining = args_remaining.intersection(set(args_remaining_policy))
            args_remaining = args_remaining.intersection(set(args_remaining_viewer))
            assert len(args_remaining) == 0, (
                f"There left unused arguments: {args_remaining}. There shouldn't be any.")
            viewer = PolicyViewer(env, policies, **args_to_pass)
            viewer.run()


    print(main.__doc__)


if __name__ == '__main__':
    logging.getLogger('').handlers = []
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    main()


