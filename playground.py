from gym.core import RewardWrapper
from matplotlib.pyplot import step
from networkx.algorithms.centrality.group import prominent_group
from networkx.classes.function import to_undirected
from torch.distributions.categorical import Categorical
from DAGs_Generator import DAGs_generate
from DAGs_Generator import plot_DAG
import gym,torch
import numpy as np




env = gym.make('MyEnv-v0')
state = env.reset()
print(state)
state,reward,done,task = env.step(0)
print(state,reward,task)
state,reward,done,task = env.step(-1)
print(state,reward,task)

# probs = torch.FloatTensor([0.05,0.1,0.2])
# dist = Categorical(probs)
# print(dist)
# probsssss = dist.probs.numpy()
# print(probsssss)
# probsssss = np.delete(probsssss,2)
# print(probsssss)
# probsssss = torch.FloatTensor(pro)
# dist = Categorical(probsssss)
# print(dist.probs.numpy())
# index = dist.sample()
# print(index.numpy())

