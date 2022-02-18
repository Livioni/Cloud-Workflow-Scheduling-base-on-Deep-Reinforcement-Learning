from collections import namedtuple
from itertools import count
import math
import os, time
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

# Parameters
gamma = 0.99                    #折扣率
render = False                  #是否显示画面       

env = gym.make("MyEnv-v0").unwrapped
state_size = env.observation_space.shape[0] #38
action_size = env.action_space.n #11

Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'next_state'])

class Actor(nn.Module):                                     #行动家网络输入为状态，输出为动作概率
    def __init__(self):
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


class Critic(nn.Module):                                    #评论家网络，输入为状态，输出为状态值函数
    def __init__(self):
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


class PPO():
    clip_param = 0.2                                        #截断系数（0.8-1.2）之间
    max_grad_norm = 0.5                                     #最大梯度范数
    ppo_update_time = 10                                    #ppo更新次数
    buffer_capacity = 500                                   #经验池 
    batch_size = 32                                         #batch大小
    actor_learning_rate = 1e-5                              #PPO对lr不敏感
    critic_learning_rate = 3e-3 

    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor()
        self.critic_net = Critic()
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.writer = SummaryWriter('models/PPOagent/exp_PPOagent')

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), self.actor_learning_rate)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), self.critic_learning_rate)
        if not os.path.exists('models/PPOagent'):
            os.makedirs('models/PPOagent/net_param')

    def select_action(self, state):                                 #选择动作
        state = torch.from_numpy(state).float().unsqueeze(0)        #转化为张量
        with torch.no_grad():                                        
            action_prob = self.actor_net(state)                     #构造动作概率
        c = Categorical(action_prob)                                #改正概率分布
        action = c.sample()                                         #采样动作
        return action.item(), action_prob[:,action.item()].item()   #返回动作和选这个动作的概率 x[:,n]表示在全部数组（维）中取第n个数据，直观来说，x[:,n]就是取所有集合的第n个数据, 

    def get_value(self, state):                                     #获得评论值
        state = torch.from_numpy(state)                             #同样转化为张量
        with torch.no_grad():                               
            value = self.critic_net(state)                          #计算状态价值
        return value.item()

    def save_param(self):
        torch.save(self.actor_net.state_dict(), 'model/PPOagent/net_param/actor_net' + str(time.time())[:10], +'.pkl')        #保存模型网络参数
        torch.save(self.critic_net.state_dict(), 'model/PPOagent/net_param/critic_net' + str(time.time())[:10], +'.pkl')

    def store_transition(self, transition):                        #修改经验池，让他不会一直增加 然后我看到了第146行的清除经验池才明白这不是DQN一般一意义的经验池
        self.buffer.append(transition)                             #这里的经验池只保留一幕的数据
        self.counter += 1

    def update(self, i_ep):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)                                    #在经验池抽取所有状态总量1000个
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)                       #在经验池抽取所有动作
        reward = [t.reward for t in self.buffer]                                                                   #抽取所所有收益
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)     #当时选择动作的概率

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + gamma * R
            Gt.insert(0, R)

        value = 0
        for ele in Gt:
            value += ele
        reward_mean = value/len(Gt)
        fangcha = 0
        for ele in Gt:
            fangcha += (ele-value)*(ele-value)
        fangcha /= len(Gt)
        reward_std = math.sqrt(fangcha)
        for t in range(len(Gt)):
            Gt[t] = (Gt[t] - reward_mean) / reward_std 

        Gt = torch.tensor(Gt, dtype=torch.float)                                                                #这里就是算好的累计收益


        # print("The agent is updateing....")
        for i in range(self.ppo_update_time):                                                                   #每一幕更新10次
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):    #在经验池中采样batch size个index
                if self.training_step % 1000 ==0:
                    print('I_ep {} ，train {} times'.format(i_ep,self.training_step))
                #with torch.no_grad():
                Gt_index = Gt[index]
                Gt_index = Gt[index].view(-1, 1)                                                            
                V = self.critic_net(state[index])
                delta = Gt_index - V                                                                            #TD_error
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(state[index]).probs.gather(1, action[index]) # new policy                #这是重新使用网络得到选择该动作的概率
                # prob = action_prob.probs.detach().numpy().flatten()[action.item()]
                ratio = (action_prob/old_action_log_prob[index])                                               #重要性采样操作
                surr1 = (ratio * advantage)                                                                      #计算重要性采样加权后的td_error
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage                #计算截断后的

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent                                #取最小值
                self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)                       #这再限制梯度？
                self.actor_optimizer.step()

                #update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.buffer[:] # clear experience

    
def main():
    agent = PPO()
    for i_epoch in range(300):
        state = env.reset()
        sum_reward = 0
        probability = {}
        if render: env.render()
        for t in count():
            # action, action_prob = agent.select_action(state)
            state = torch.from_numpy(state).float().unsqueeze(0)        #转化为张量
            with torch.no_grad():                                        
                action_prob = agent.actor_net(state)                     #构造动作概率
            for j in range(11):
                probability[j] = action_prob.probs.detach().numpy().flatten()[j]
            action = action_prob.sample()#采样当前动作
            state,reward,done,info = env.step(action.item()-1)
            prob = action_prob.probs.numpy().flatten()[action.item()]
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
                prob = action_prob.probs.numpy().flatten()[action.item()]
                state,reward,done, info = env.step(action.item()-1)#输入step的都是
            next_state, reward, done, _ = state, reward, done, info
            trans = Transition(state, action, prob, reward, next_state)
            if render: env.render()
            agent.store_transition(trans)
            state = next_state
            sum_reward += reward
            if done :
                time = state[0]
                if len(agent.buffer) >= agent.batch_size:agent.update(i_epoch)                  #这里要每一幕至少运行32次才能训练
                print("Episode:{}   Sum reward:{}, Makespan:{:.3f}".format(i_epoch,sum_reward,time))
                agent.writer.add_scalar('info/Reward', sum_reward, global_step=i_epoch)
                agent.writer.add_scalar('info/makespan', time, global_step=i_epoch)
                break

if __name__ == '__main__':
    main()
    print("end")
