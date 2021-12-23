from networkx.classes.function import to_undirected
from DAGs_Generator import DAGs_generate
from DAGs_Generator import plot_DAG
import gym
import numpy as np


env = gym.make('MyEnv-v0')
state = env.reset()
print(state)

