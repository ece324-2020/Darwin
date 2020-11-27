from agents.default import policies
from envs.default import darwin_envs
from wrappers.discrete import DiscretizedObservationWrapper

def load_env(env_name):
    # Wrapper for any other steps to initialize env
    try:
        env = darwin_envs[env_name]
        # Discretize observations
        # env = DiscretizedObservationWrapper(
        #     env
        # )
    except KeyError:
        raise KeyError('Specified environment does not exist.')

    return env


def load_policy(policy_name, env):
    # Wrapper for any other steps to initialize env
    try:
        policy = policies[policy_name](env)
    except KeyError:
        raise KeyError('Specified policy does not exist.')

    return policy
