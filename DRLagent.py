import gym, os
from itertools import count
from numpy import nested_iters
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter(comment='Cartpole Reward Record')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("MyEnv-v0").unwrapped

state_size = env.observation_space.shape[0] #38
action_size = env.action_space.n #11
lr = 0.001 #学习率 
n_iters=1
sum_reward = 0
episode_durations = []        

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


def compute_returns(next_value, rewards, gamma=0.99):#计算回报
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):#还是REINFORCE方法
        R = rewards[step] + gamma * R  #gamma是折扣率
        returns.insert(0, R)
    return returns


def trainIters(actor, critic, n_iters):
    optimizerA = optim.Adam(actor.parameters(),lr)
    optimizerC = optim.Adam(critic.parameters(),lr)
    for iter in range(n_iters):
        state = env.reset()
        log_probs = []
        sum_reward = 0 
        values = []
        rewards = []
        for i in count():
            # env.render()
            state = torch.FloatTensor(state).to(device)
            dist, value = actor(state), critic(state) #dist得出动作概率分布，value得出当前动作价值函数
            print(state)
            action = dist.sample()#采样当前动作 
            state,reward,done,info = env.step(action.numpy()-1)
            while (info == False):
                action = dist.sample()#采样当前动作 
                state,reward,done, info = env.step(action.numpy()-1)#输入step的都是
                # print("take an action:",action.numpy()-1)
            next_state, reward, done, _ = state, reward, done, info
            log_prob = dist.log_prob(action).unsqueeze(0)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))

            state = next_state

            if done:
                episode_durations.append(i + 1)
                print('Iteration: {}, Score: {}'.format(iter, i + 1))
                break


        next_state = torch.FloatTensor(next_state).to(device)
        next_value = critic(next_state)
        returns = compute_returns(next_value, rewards)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()#这个使用REINFORCE
        critic_loss = advantage.pow(2).mean()

        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()
        #绘制曲线
        # for data in range(n_iters):
        #     writer.add_scalar('Reward',sum_reward,data)   
        # writer.close()
    torch.save(actor, 'models/actor.pkl')
    torch.save(critic, 'models/critic.pkl')
    env.close()



if __name__ == '__main__':
    # if os.path.exists('models/actor.pkl'):
    #     actor = torch.load('models/actor.pkl')
    #     print('Actor Model loaded')
    # else:
    actor = Actor(state_size, action_size).to(device)
    # if os.path.exists('models/critic.pkl'):
    #     critic = torch.load('models/critic.pkl')
    #     print('Critic Model loaded')
    # else:
    critic = Critic(state_size, action_size).to(device)
    trainIters(actor, critic, n_iters=n_iters)

