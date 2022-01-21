from os import stat
from gym.core import RewardWrapper
from matplotlib.pyplot import step
from networkx.algorithms.centrality.group import prominent_group
from networkx.classes.function import info, to_undirected
from torch.distributions.categorical import Categorical
from DAGs_Generator import DAGs_generate
from DAGs_Generator import plot_DAG
import gym,torch
import numpy as np




env = gym.make('MyEnv-v0')
state = env.reset()
print(state)
state,reward,done,info = env.step(0)
print(state,info)
state,reward,done,info = env.step(1)
print(state,info)
state,reward,done,info = env.step(-1)
print(state,info)

