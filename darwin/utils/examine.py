import os.path as p
import click

from env_parsing import parse_kwargs

from mujoco_worldgen.util.envs import EnvViewer
from mujoco_worldgen.util.envs import examine_env
from policies.load_policy import load_policy

ENV_DIR = p.abspath(p.join(p.dirname(__file__), '..', 'darwin_envs'))


@click.command()
@click.argument('name')
@click.argument('policy', required=False, default=None)
@click.argument('kwargs', nargs=-1, required=False)
def examine(name, policy, kwargs):
    """
    Visualize an environment with the path specified
    """
    if not policy:
        examine_env(name, parse_kwargs(kwargs), envs_dir=ENV_DIR, env_viewer=EnvViewer)
    else:
        env, _ = 
        load_policy(policy)


if __name__ == "__main__":
    examine()
