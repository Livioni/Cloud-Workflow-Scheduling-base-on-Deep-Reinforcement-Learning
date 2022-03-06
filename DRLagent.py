import gym, os, math, random
import numpy as np
from itertools import count
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(comment='Workflow scheduler Reward Record')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("MyEnv-v0").unwrapped

state_size = env.observation_space.shape[0]  # 38
action_size = env.action_space.n  # 11
lr = 0.0001  # 学习率
n_iters = 50000
sum_reward = 0
time_durations = []


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


def compute_returns(next_value, rewards, gamma=0.99):  # 计算回报
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):  # 还是REINFORCE方法
        R = rewards[step] + gamma * R  # gamma是折扣率
        returns.insert(0, R)

    reward_mean = np.mean(returns)
    reward_std = np.std(returns)
    for t in range(len(returns)):
        returns[t] = (returns[t] - reward_mean) / reward_std
    return returns


def trainIters(actor, critic, n_iters):
    optimizerA = optim.Adam(actor.parameters(), lr)
    optimizerC = optim.Adam(critic.parameters(), lr)
    for iter in range(n_iters):
        state = env.reset()
        log_probs = []
        sum_reward = 0
        time = 0
        values = []
        rewards = []
        probability = {}
        probability_list = []
        total_makespan = 0
        average_makespan = 0
        for i in count():
            # env.render()
            state = torch.FloatTensor(state).to(device)
            dist, value = actor(state), critic(state)  # dist得出动作概率分布，value得出当前动作价值函数
            for j in range(action_size):
                probability[j] = dist.probs.detach().numpy()[j]
            action = dist.sample()  # 采样当前动作
            state, reward, done, info = env.step(action.numpy() - 1)
            while (info[0] == False):  # 重采样
                probability[action.item()] = 0
                probability_list = [probs for probs in probability.values()]
                probs = torch.FloatTensor(probability_list)
                dist_copy = Categorical(probs)
                for j in range(len(dist_copy.probs)):
                    probability_list[j] = dist_copy.probs[j].item()
                probs = torch.FloatTensor(probability_list)
                dist_1 = Categorical(probs)
                action = dist_1.sample()  # 采样当前动作
                state, reward, done, info = env.step(action.numpy() - 1)  # 输入step的都是
            next_state, reward, done, _ = state, reward, done, info
            log_prob = dist.log_prob(action).unsqueeze(0)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            state = next_state
            sum_reward += reward
            if done:
                time = state[0]
                time_durations.append(time)
                writer.add_scalar('info/Makespan', time, global_step=iter + 1)
                writer.add_scalar('info/Sum_reward', sum_reward, global_step=iter + 1)
                print('Episode: {}, Reward: {:.3f}, Makespan: {:.3f}s'.format(iter + 1, sum_reward, time))
                break

        if (n_iters % 500 == 0):
            torch.save(actor, 'models/ACagent/actor.pkl')
            torch.save(critic, 'models/ACagent/critic.pkl')

        next_state = torch.FloatTensor(next_state).to(device)
        next_value = critic(next_state).detach().numpy()
        returns = compute_returns(next_value, rewards)
        returns = torch.tensor(np.array(returns), dtype=torch.float, device=device)

        log_probs = torch.cat(log_probs)
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()  # 这个使用REINFORCE  加上负号表示梯度上升
        writer.add_scalar('Loss/actor_loss', actor_loss, global_step=iter + 1)

        critic_loss = advantage.pow(2).mean()
        writer.add_scalar('Loss/critic_loss', critic_loss, global_step=iter + 1)

        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()
        # 绘制曲线

    for times in time_durations:
        total_makespan += times

    average_makespan = total_makespan / n_iters
    print(average_makespan)
    torch.save(actor, 'models/ACagent/actor.pkl')
    torch.save(critic, 'models/ACagent/critic.pkl')
    env.close()
    writer.close()


def show_makespan():
    plt.figure(3)
    plt.grid()
    plt.clf()
    durations_t = torch.FloatTensor(time_durations)
    plt.title('Makespan of each episode')
    plt.xlabel('Episode')
    plt.ylabel('Makespan(s)')
    plt.plot(durations_t.numpy())
    # 平滑处理
    x = np.linspace(1, n_iters, n_iters)
    y = savgol_filter(durations_t.numpy(), 99, 3, mode='nearest')
    plt.plot(x, y, 'k', label='savgol')
    plt.savefig("Makespan.png", format="PNG")
    plt.show()


if __name__ == '__main__':
    print("--------------------------------------------------------------------------------------------")

    print("状态空间维数 : ", state_size)
    print("动作空间维数 : ", action_size)

    print("--------------------------------------------------------------------------------------------")
    if os.path.exists('models/ACagent/actor.pkl'):
        actor = torch.load('models/ACagent/actor.pkl')
        print('Actor Model loaded')
    else:
        actor = Actor(state_size, action_size).to(device)
        # actor.apply(weights_init)
    if os.path.exists('models/ACagent/critic.pkl'):
        critic = torch.load('models/ACagent/critic.pkl')
        print('Critic Model loaded')
    else:
        critic = Critic(state_size, action_size).to(device)
        # critic.apply(weights_init)
    trainIters(actor, critic, n_iters=n_iters)
    show_makespan()
