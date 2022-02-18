from time import time
import gym
from matplotlib.image import imread
from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import xlwt

env = gym.make("MyEnv-v0").unwrapped
state_size = env.observation_space.shape[0] #38
action_size = env.action_space.n #11
test_order = 100
sum_reward = 0
time_durations = []        

def initial_excel():
    global worksheet,workbook
    # xlwt 库将数据导入Excel并设置默认字符编码为ascii
    workbook = xlwt.Workbook(encoding='ascii')
    #添加一个表 参数为表名
    worksheet = workbook.add_sheet('makespan')
    # 生成单元格样式的方法
    # 设置列宽, 3为列的数目, 12为列的宽度, 256为固定值
    for i in range(3):
        worksheet.col(i).width = 256 * 12
    # 设置单元格行高, 25为行高, 20为固定值
    worksheet.row(1).height_mismatch = True
    worksheet.row(1).height = 20 * 25
    # 保存excel文件
    workbook.save('data/makespan_AC.xls') 


class Actor(nn.Module): #策略网络
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 40)
        self.linear2 = nn.Linear(40,40)
        self.linear3 = nn.Linear(40, self.action_size)

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
        self.linear1 = nn.Linear(self.state_size, 40)
        self.linear2 = nn.Linear(40, 40)
        self.linear3 = nn.Linear(40, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value #输出状态值函数

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
            state = next_state
            sum_reward += reward
            if done:
                time = state[0]
                time_to_write = round(float(time),3)
                worksheet.write(o, 0, time_to_write)
                workbook.save('data/makespan_AC.xls') 
                print("Makespan: {:.3f} s".format(time))
                print('Reward: {:.3f}'.format(sum_reward))
                break
            # img = imread('DAG.png')
            # plt.imshow(img)
            # plt.axis('off') # 不显示坐标轴
            # plt.show()



if __name__ == '__main__':
    actor = torch.load('models/ACagent/actor.pkl')
    critic = torch.load('models/ACagent/critic.pkl')
    initial_excel()
    test(actor, critic)  