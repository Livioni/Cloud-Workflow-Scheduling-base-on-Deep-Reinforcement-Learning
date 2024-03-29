import os
from datetime import datetime
import gym
import numpy as np
import torch
import torch.nn as nn
import xlwt
from torch.distributions import Categorical, MultivariateNormal


def initial_excel():
    global worksheet, workbook
    # xlwt 库将数据导入Excel并设置默认字符编码为ascii
    workbook = xlwt.Workbook(encoding='ascii')
    # 添加一个表 参数为表名
    worksheet = workbook.add_sheet('resources usage')
    # 生成单元格样式的方法
    # 设置列宽, 3为列的数目, 12为列的宽度, 256为固定值
    for i in range(3):
        worksheet.col(i).width = 256 * 12
    # 设置单元格行高, 25为行高, 20为固定值
    worksheet.row(1).height_mismatch = True
    worksheet.row(1).height = 20 * 25
    worksheet.write(0, 0, 'time')
    worksheet.write(0, 1, 'CPU usage(%)')
    worksheet.write(0, 2, 'Memory usage(%)')
    worksheet.write(0, 3, 'time1')
    worksheet.write(0, 4, 'CPU(%)')
    worksheet.write(0, 5, 'Memory(%)')
    # 保存excel文件
    workbook.save('data/res_monitor.xls')


print("============================================================================================")
####### initialize environment hyperparameters ######
env_name = "clusterEnv-v0"  # 定义自己的环境名称
max_ep_len = 10000  # max timesteps in one episode
total_test_episodes = 1  # total num of testing episodes

################ PPO hyperparameters ################

K_epochs = 80  # update policy for K epochs in one PPO update
eps_clip = 0.2  # clip parameter for PPO
gamma = 0.99  # discount factor

lr_actor = 0.0003  # learning rate for actor network
lr_critic = 0.001  # learning rate for critic network

#####################################################

print("Testing environment name : " + env_name)

env = gym.make(env_name).unwrapped

# state space dimension # action space dimension
state_dim, action_dim = env.return_dim_info()

################### checkpointing ###################

run_num_pretrained = 'resourceTest'  #### change this to prevent overwriting weights in same env_name folder

directory = "runs/PPO_preTrained"
if not os.path.exists(directory):
    os.makedirs(directory)

directory = directory + '/' + 'clusterEnv-v0' + '/'
if not os.path.exists(directory):
    os.makedirs(directory)

checkpoint_path = directory + "PPO_clusterEnv-v0_{}.pth".format(run_num_pretrained)
print("save checkpoint path : " + checkpoint_path)

#####################################################


############# print all hyperparameters #############

print("--------------------------------------------------------------------------------------------")

print("资源用量监视")

print("--------------------------------------------------------------------------------------------")

print("状态空间维数 : ", state_dim)
print("动作空间维数 : ", action_dim)

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')

if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

print("============================================================================================")

line = 1
flag = 0


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        # global flag,line
        probability = {}
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        for j in range(action_dim):
            probability[j] = dist.probs.detach()[j]  # 记录当前动作概率分布
        action = dist.sample()
        state, reward, done, info = env.step(action.item() - 1)
        while (info[0] == False):  # 重采样
            probability[action.item()] = 0
            probability_list = [probs for probs in probability.values()]
            probs = torch.FloatTensor(probability_list)
            dist_copy = Categorical(probs)
            for j in range(len(dist_copy.probs)):
                probability_list[j] = dist_copy.probs[j].item()
            probs = torch.FloatTensor(probability_list)
            dist_1 = Categorical(probs)
            action = dist_1.sample().to(device)  # 采样当前动作
            # if action.item() == 0:
            #     time, cpu_usage, memory_usage = env.return_res_usage()
            #     worksheet.write(line, 1, str(100-cpu_usage)+'%')
            #     worksheet.write(line, 2, str(100-memory_usage)+'%')   
            #     flag = 1         
            #     line += 1
            state, reward, done, info = env.step(action.item() - 1)  # 输入step的都是
        action_logprob = dist.log_prob(action).unsqueeze(0)
        return action.detach(), action_logprob.detach(), state, reward, done, info

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()  # 经验池

        self.policy = ActorCritic(state_dim, action_dim).to(device)  # AC策略
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim).to(device)  # AC策略old网络
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            self.buffer.states.append(state)
            action, action_logprob, state, reward, done, info = self.policy_old.act(state)

        self.buffer.actions.append(action)  # 保存动作
        self.buffer.logprobs.append(action_logprob)  # 保存动作概率
        return state, reward, done, info

    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


def test():
    ################# testing procedure ################
    # initialize a PPO agent
    global flag
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)
    # track total testing time

    ppo_agent.load(checkpoint_path)
    print("Network ID:", run_num_pretrained)
    print('PPO agent has been loaded!')

    # GCN test
    state = env.reset()
    for t in range(1, max_ep_len + 1):
        # select action with policy
        new_state, reward, done, info = ppo_agent.select_action(state)
        # saving reward and is_terminals
        # break; if the episode is over
        if flag == 1:
            time, cpu_usage, memory_usage = env.return_res_usage()
            worksheet.write(line - 1, 0, time)
            flag = 0
        # 记录资源使用率
        state = new_state
        if done:
            print("makesspan:", state[0])
            break
    ppo_agent.buffer.clear()
    # workbook.save('data/res_monitor.xls')
    env.close()


if __name__ == '__main__':
    # initial_excel()
    test()
