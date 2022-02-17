from cmath import inf
from time import time
from traceback import print_tb
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
import xlwt

env = gym.make("MyEnv-v0").unwrapped

def find_shortest_job(state):
    '''
    寻找shortest的job
    :param state: 当前状态 
    :return: node的后续节点id列表
    '''
    ready_job_list = state[3:13].tolist()
    min = inf
    for ele in ready_job_list:
        if ele != -1:
            min = ele if ele < min else min
    shortest_ind = ready_job_list.index(min)
    return shortest_ind + 3

def sjf():
    state = env.reset()
    print(state)
    shortest_ind = find_shortest_job(state)
    print(state[shortest_ind])


def test(actor, critic):
    global worksheet,workbook
    for o in range(test_order):
        state = env.reset()
        sum_reward = 0 
        time = 0
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
            state = next_state
            sum_reward += reward
            if done:
                time = state[0]
                time_to_write = round(float(time),3)
                worksheet.write(o, 0, time_to_write)
                workbook.save('makespan.xls') 
                print("Makespan: {} s".format(time))
                print('Reward: {}'.format(sum_reward))
                break
            # img = imread('DAG.png')
            # plt.imshow(img)
            # plt.axis('off') # 不显示坐标轴
            # plt.show()



if __name__ == '__main__':
    sjf()  