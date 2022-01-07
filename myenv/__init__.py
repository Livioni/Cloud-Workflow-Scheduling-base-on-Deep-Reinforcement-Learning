from gym.envs.registration import register

register(
    id='MyEnv-v0',
    entry_point='myenv.myEnv:MyEnv',
)