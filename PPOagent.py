import os
from datetime import datetime

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(comment='Workflow scheduler Reward Record')
print("============================================================================================")
####### initialize environment hyperparameters ######
env_name = "MyEnv-v0"  # 定义自己的环境名称
max_ep_len = 1000  # max timesteps in one episode
max_training_timesteps = int(3e5)  # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len / 2   # print avg reward in the interval (in num timesteps)
save_model_freq = int(5e3)  # save model frequency (in num timesteps)

#####################################################

## Note : print frequencies should be > than max_ep_len

################ PPO hyperparameters ################

update_timestep = max_ep_len  # update policy every n timesteps
K_epochs = 80  # update policy for K epochs in one PPO update

eps_clip = 0.2  # clip parameter for PPO
gamma = 0.99  # discount factor

lr_actor = 0.0003  # learning rate for actor network
lr_critic = 0.001  # learning rate for critic network

#####################################################

print("training environment name : " + env_name)

env = gym.make(env_name).unwrapped

# state space dimension action space dimension
state_dim,action_dim = env.return_dim_info()

################### checkpointing ###################

run_num_pretrained = '30'  #### change this to prevent overwriting weights in same env_name folder

directory = "runs/PPO_preTrained"
if not os.path.exists(directory):
    os.makedirs(directory)

directory = directory + '/' + env_name + '/'
if not os.path.exists(directory):
    os.makedirs(directory)

checkpoint_path = directory + "PPO_{}_{}.pth".format(env_name, run_num_pretrained)
print("save checkpoint path : " + checkpoint_path)

#####################################################


############# print all hyperparameters #############

print("--------------------------------------------------------------------------------------------")

print("最大步数 : ", max_training_timesteps)
print("每一幕的最大步数 : ", max_ep_len)

print("模型保存频率 : " + str(save_model_freq) + " timesteps")
print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")

print("--------------------------------------------------------------------------------------------")

print("状态空间维数 : ", state_dim)
print("动作空间维数 : ", action_dim)

print("--------------------------------------------------------------------------------------------")

print("初始化离散动作空间策略")

print("--------------------------------------------------------------------------------------------")

print("PPO 更新频率 : " + str(update_timestep) + " timesteps")
print("PPO K epochs : ", K_epochs)
print("PPO epsilon clip : ", eps_clip)
print("discount factor (gamma) : ", gamma)

print("--------------------------------------------------------------------------------------------")

print("optimizer learning rate actor : ", lr_actor)
print("optimizer learning rate critic : ", lr_critic)

#####################################################

print("============================================================================================")

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
        self.record = 0

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
            # for j in range(update_timestep):
            #     writer.add_scalar('Loss/PPO_loss', loss[j], global_step=self.record)
            #     self.record += 1
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


def train():
    ################# training procedure ################
    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)
    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    if not os.path.exists(checkpoint_path):
        print('Network Initilized.')
    else:
        ppo_agent.load(checkpoint_path)
        print("PPO has been loaded!")

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0


    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len + 1):

            # select action with policy
            state, reward, done, info = ppo_agent.select_action(state)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)  # 保存收益
            ppo_agent.buffer.is_terminals.append(done)  # 保存是否完成一幕

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                print('Network updating.')
                ppo_agent.update()

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)
                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                        print_avg_reward))
                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                time = state[0]
                time_to_write = round(float(time), 3)
                writer.add_scalar('info/PPO_makespan', time_to_write, global_step=i_episode)
                writer.add_scalar('info/PPO_Sum_reward', current_ep_reward, global_step=i_episode)
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        i_episode += 1
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    train()
