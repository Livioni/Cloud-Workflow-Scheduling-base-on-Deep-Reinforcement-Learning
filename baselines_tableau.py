import random
import xlsxwriter
import torch.nn as nn
import gym
from itertools import count
import torch
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np


class Actor(nn.Module):  # 策略网络
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 40)
        self.linear2 = nn.Linear(40, 40)
        self.linear3 = nn.Linear(40, self.action_size)

    def forward(self, state):
        output = torch.sigmoid(self.linear1(state))
        output = self.linear2(output)
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution  # 输出动作概率分布


class Critic(nn.Module):  # 状态值函数网络
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 40)
        self.linear2 = nn.Linear(40, 40)
        self.linear3 = nn.Linear(40, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value  # 输出状态值函数


def check_res_Tetris(state):
    job_cpu_demand = state[33:63]
    job_memory_demand = state[63:93]
    cpu_res = state[1]
    memory_res = state[2]
    for i in range(len(job_cpu_demand)):
        if ((job_cpu_demand[i] == -1.0) and (job_memory_demand[i] == -1.0)):
            continue
        else:
            if (job_cpu_demand[i] > cpu_res or job_memory_demand[i] > memory_res):
                job_cpu_demand[i] = -1.0
                job_memory_demand[i] = -1.0
            else:
                continue
    state[33:63] = job_cpu_demand
    state[63:93] = job_memory_demand
    return np.array(state, dtype=np.float32)


def alignment_score(state):
    job_cpu_demand = state[33:63]
    job_memory_demand = state[63:93]
    cpu_res = state[1]
    memory_res = state[2]
    alignment_score = cpu_res * job_cpu_demand + memory_res * job_memory_demand
    if all(map(lambda x: x < 0, alignment_score)):
        return -1
    else:
        return np.where(alignment_score == np.max(alignment_score))[0][0]


def find_shortest_job(state):
    '''
    寻找shortest的job
    :param state: 当前状态 
    :return: shortest job在[0:9]中的索引
    '''
    ready_job_list = state[3:33].tolist()
    min = 999999
    for ele in ready_job_list:
        if ele != -1:
            min = ele if ele < min else min
    shortest_ind = ready_job_list.index(min)
    return shortest_ind


def check_res(state):
    '''
    判断当前机器是否还可以装载
    :param state: 当前状态 
    :return: bool值 是否还可以装载
    '''
    job_duration = state[3:33].tolist()
    job_cpu_demand = state[33:63].tolist()
    job_memory_demand = state[63:93].tolist()
    cpu_res = state[1]
    memory_res = state[2]
    flag = False
    for i in range(len(job_duration)):
        if ((job_cpu_demand[i] == -1.0) and (job_memory_demand[i] == -1.0)):
            continue
        else:
            flag = True if (job_cpu_demand[i] < cpu_res and job_memory_demand[i] < memory_res) else False
            if flag == True:
                break
    return flag


def check_ready(state, index):
    '''
    判断当前机器是否还可以装载任务index
    :param state: 当前状态 
    :param index: 查询的任务index 
    :return: bool值 是否还可以装载
    '''
    job_cpu_demand = state[33:63].tolist()
    job_memory_demand = state[63:93].tolist()
    cpu_res = state[1]
    memory_res = state[2]
    return True if (job_cpu_demand[index] < cpu_res and job_memory_demand[index] < memory_res) else False


def test(actor, test_order):
    global worksheet, workbook
    print("AC")
    makespans = []
    line = 1
    for o in range(1, test_order + 1):
        state = env.reset()
        sum_reward = 0
        time = 0
        probability = {}
        probability_list = []
        for i in count():
            # env.render()
            state = torch.FloatTensor(state)
            dist = actor(state)  # dist得出动作概率分布，value得出当前动作价值函数
            for i in range(action_size):
                probability[i] = dist.probs.detach().numpy()[i]
            action = dist.sample()  # 采样当前动作
            state, reward, done, info = env.step(action.numpy() - 1)
            while (info[0] == False):  # 重采样
                probability[action.item()] = 0
                probability_list = [probs for probs in probability.values()]
                probs = torch.FloatTensor(probability_list)
                dist_copy = Categorical(probs)
                for i in range(len(dist_copy.probs)):
                    probability_list[i] = dist_copy.probs[i].item()
                probs = torch.FloatTensor(probability_list)
                dist_1 = Categorical(probs)
                action = dist_1.sample()  # 采样当前动作
                state, reward, done, info = env.step(action.numpy() - 1)  # 输入step的都是
            next_state, reward, done, _ = state, reward, done, info
            state = next_state
            sum_reward += reward
            if done:
                time = state[0]
                makespans.append(time)
                # print("Makespan: {:.3f} s".format(time))
                if o % auto_save == 0:
                    average_makespan = np.mean(makespans)
                    worksheet.write(line, 1, average_makespan)
                    print('AC : Episode: {}, Reward: {:.3f}, Makespan: {:.3f}s'.format(line * auto_save, sum_reward,
                                                                                       average_makespan))
                    line += 1
                    makespans = []
                break


def tetris(n_iters):
    print("Tetris")
    makespans = []
    line = 1
    for iter in range(1, n_iters + 1):
        state = env.reset()
        sum_reward = 0  # 记录每一幕的reward
        time = 0  # 记录makespan
        for i in count():
            valid_state = check_res_Tetris(state)
            action = alignment_score(valid_state)
            next_state, reward, done, info = env.step(action)
            sum_reward += reward
            state = next_state
            if done:
                time = state[0]
                makespans.append(time)
                # print("Episode:",iter,"makespan:",time)
                if iter % auto_save == 0:
                    average_makespan = np.mean(makespans)
                    worksheet.write(line + 100, 1, average_makespan)
                    print('Tetris : Episode: {}, Reward: {:.3f}, Makespan: {:.3f}s'.format(line * auto_save, sum_reward,
                                                                                           average_makespan))
                    line += 1
                    makespans = []
                break


def sjf(n_iters):
    print("SJF")
    makespans = []
    line = 1
    for iter in range(1, n_iters + 1):
        state = env.reset()
        sum_reward = 0  # 记录每一幕的reward
        time = 0  # 记录makespan
        for i in count():
            if (check_res(state)):
                preaction = find_shortest_job(state)
                if check_ready(state, preaction):
                    action = preaction
                else:
                    action = -1
            else:
                action = -1
            next_state, reward, done, info = env.step(action)
            sum_reward += reward
            state = next_state
            if done:
                time = state[0]
                time_to_write = round(float(time), 3)
                makespans.append(time_to_write)
                if iter % auto_save == 0:
                    average_makespan = np.mean(makespans)
                    worksheet.write(line + 200, 1, average_makespan)
                    print('SJF : Episode: {}, Reward: {:.3f}, Makespan: {:.3f}s'.format(line * auto_save, sum_reward,
                                                                                        average_makespan))
                    line += 1
                    makespans = []
                break


def randomagent(n_iters):
    print("random")
    makespans = []
    line = 1
    for iter in range(1, n_iters + 1):
        state = env.reset()
        sum_reward = 0  # 记录每一幕的reward
        time = 0  # 记录makespan
        for i in count():
            action = random.choice(range(action_size)) - 1
            state, reward, done, info = env.step(action)
            while (info[0] == False):
                action = random.choice(range(action_size)) - 1
                state, reward, done, info = env.step(action)  # 输入step的都是
            next_state, reward, done, _ = state, reward, done, info
            sum_reward += reward
            state = next_state
            if done:
                time = state[0]
                makespans.append(time)
                if iter % auto_save == 0:
                    average_makespan = np.mean(makespans)
                    worksheet.write(line + 300, 1, average_makespan)
                    print('Random : Episode: {}, Reward: {:.3f}, Makespan: {:.3f}s'.format(line * auto_save, sum_reward,
                                                                                           average_makespan))
                    line += 1
                    makespans = []
                break


if __name__ == '__main__':
    n = 6  # 有多少个方法对比
    # Create an new Excel file and add a worksheet.
    workbook = xlsxwriter.Workbook('data/Makespans50.xlsx')
    worksheet = workbook.add_worksheet()
    # Widen the first column to make the text clearer.
    worksheet.set_column('A:A', 15)
    # Add a bold format to use to highlight cells.
    bold = workbook.add_format({'bold': True})
    # Write some simple text.
    worksheet.write('A1', '序号')
    worksheet.write('B1', 'Makespan(s)')
    worksheet.write('C1', '方法')
    worksheet.write('D1', 'DAG大小')
    for line in range(0, n):
        for i in range(100):
            worksheet.write(i + 1 + line * 100, 0, i + 1)
    for i in range(100):
        worksheet.write(i + 1, 2, 'Actor-Critic')
    for i in range(100, 200):
        worksheet.write(i + 1, 2, 'Tetris')
    for i in range(200, 300):
        worksheet.write(i + 1, 2, 'SJF')
    for i in range(300, 400):
        worksheet.write(i + 1, 2, 'Random')
    for i in range(400, 500):
        worksheet.write(i + 1, 2, 'PPO')
    for i in range(500, 600):
        worksheet.write(i + 1, 2, 'MCTS')
    for i in range(100 * n):
        worksheet.write(i + 1, 3, 'n=50')

    env = gym.make("clusterEnv-v0").unwrapped
    state_size, action_size = env.return_dim_info()
    auto_save = 10
    test_order = 100 * auto_save
    sum_reward = 0
    time_durations = []
    actor = torch.load('models/ACagent/actor.pkl')
    tetris(test_order)
    test(actor, test_order)
    sjf(test_order)
    randomagent(test_order)
    workbook.close()
