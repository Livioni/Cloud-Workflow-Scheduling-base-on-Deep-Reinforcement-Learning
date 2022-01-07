from gym.core import RewardWrapper
from matplotlib.pyplot import step
from networkx.algorithms.centrality.group import prominent_group
from networkx.classes.function import info, to_undirected
from torch.distributions.categorical import Categorical
from DAGs_Generator import DAGs_generate
from DAGs_Generator import plot_DAG
import gym,torch
import numpy as np




# env = gym.make('MyEnv-v0')
# state = env.reset()
# print(state)
# state,reward,done,info = env.step(0)
# print(state,reward,done,info)
# state,reward,done,task = env.step(-1)
# print(state,reward,done,info)
probability = {}
probs = torch.FloatTensor([0.05,0.1,0.2])
dist = Categorical(probs)
for i in range(len(dist.probs.numpy())):
    probability[i] = dist.probs.numpy()[i]
print(probability)
probability_list = [probs for probs in probability.values()]
print(probability_list)
probability_list[0] = 0 
probs = torch.FloatTensor(probability_list)
dist_1 = Categorical(probs)
print(len(dist_1.probs))
