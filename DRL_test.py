from time import time
import gym, os,math
from matplotlib.image import imread
import numpy as np
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt

env = gym.make("MyEnv-v0").unwrapped
state_size = env.observation_space.shape[0] #38
action_size = env.action_space.n #11
lr = 0.0001 #学习率 
sum_reward = 0
time_durations = []        

class Actor(nn.Module): #策略网络
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution #输出动作概率分布


class Critic(nn.Module): #状态值函数网络
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value #输出状态值函数

def test(actor, critic):
    state = env.reset()
    log_probs = []
    sum_reward = 0 
    time = 0
    values = []
    rewards = []
    probability = {}
    probability_list = []
    for i in count():
        # env.render()
        state = torch.FloatTensor(state)
        dist, value = actor(state), critic(state) #dist得出动作概率分布，value得出当前动作价值函数
        for i in range(11):
            probability[i] = dist.probs.detach().numpy()[i]
        action = dist.sample()#采样当前动作
        state,reward,done,info = env.step(action.numpy()-1)
        while (info == False):                                              #重采样
            probability[action.item()] = 0
            probability_list = [probs for probs in probability.values()]
            probs = torch.FloatTensor(probability_list)
            dist_copy = Categorical(probs) 
            for i in range(len(dist_copy.probs)):
                probability_list[i] = dist_copy.probs[i].item()
            probs = torch.FloatTensor(probability_list)
            dist_1 = Categorical(probs)
            action = dist_1.sample()#采样当前动作 
            state,reward,done, info = env.step(action.numpy()-1)#输入step的都是
        next_state, reward, done, _ = state, reward, done, info
        log_prob = dist.log_prob(action).unsqueeze(0)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.tensor([reward], dtype=torch.float))         
        state = next_state
        sum_reward += reward
        if done:
            time = state[0]
            print("Makespan: {} s".format(time))
            print('Reward: {}'.format(sum_reward))
            break
    img = imread('DAG.png')
    plt.imshow(img)
    plt.axis('off') # 不显示坐标轴
    plt.show()


if __name__ == '__main__':

    actor = torch.load('models/actor.pkl')
    critic = torch.load('models/critic.pkl')
    test(actor, critic)  