import gym
from gym.envs.registration import register, registry


def register_cluster_env():
    env_id = "clusterEnv-v0"
    if env_id in registry:
        return

    version = tuple(int(part) for part in gym.__version__.split(".")[:2])
    registration_options = {}
    if version >= (0, 25):
        registration_options["disable_env_checker"] = True

    register(
        id=env_id,
        entry_point="Envs.clusterEnv:clusterEnv",
        **registration_options,
    )
