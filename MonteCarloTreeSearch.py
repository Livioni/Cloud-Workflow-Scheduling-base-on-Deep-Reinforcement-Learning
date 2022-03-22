import gym,torch,copy,os
from sympy import true
import torch.nn as nn
from datetime import datetime
import numpy as np
from torch.distributions import Categorical, MultivariateNormal

env = gym.make("clusterEnv-v0").unwrapped
state_dim,action_dim = env.return_dim_info()
################### checkpointing ###################
run_num_pretrained = 5
directory = "runs/PPO_preTrained"
if not os.path.exists(directory):
    os.makedirs(directory)
directory = directory + '/' + 'clusterEnv-v0' + '/'
if not os.path.exists(directory):
    os.makedirs(directory)
checkpoint_path = directory + "PPO_clusterEnv-v0_{}.pth".format(run_num_pretrained)
#####################################################


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
        # action = np.argmax(dist.probs)
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
            action = dist_1.sample()  # 采样当前动作
            # action = np.argmax(dist_1.probs)  # 采样当前动作
            state, reward, done, info = env.step(action.item() - 1)  # 输入step的都是
        action_logprob = dist.log_prob(action).unsqueeze(0)
        return action.detach(), action_logprob.detach(), state, reward, done, info

class PPO:
    def __init__(self, state_dim, action_dim):

        self.policy = ActorCritic(state_dim, action_dim)  # AC策略
        self.policy_old = ActorCritic(state_dim, action_dim)  # AC策略old网络
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            action, action_logprob, state, reward, done, info = self.policy_old.act(state)

        return state, reward, done, info


    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

def read_current_state():
    '''
    读取当前env的状态
    :return: 当前env的状态
    '''
    state = copy.deepcopy(env.state)
    ready_list = copy.deepcopy(env.ready_list)
    done_job = copy.deepcopy(env.done_job)
    tasks =copy.deepcopy( env.tasks)
    wait_duration = copy.deepcopy(env.wait_duration)
    cpu_demand = copy.deepcopy(env.cpu_demand)
    memory_demand = copy.deepcopy(env.memory_demand)
    tasks_remaing_time = copy.deepcopy(env.tasks_remaing_time)
    cpu_res = env.cpu_res
    memory_res = env.memory_res
    return state,ready_list,done_job,tasks,wait_duration,cpu_demand,memory_demand,tasks_remaing_time,cpu_res,memory_res

def load_current_state(state,ready_list,done_job,tasks,wait_duration,cpu_demand,memory_demand,tasks_remaing_time,cpu_res,memory_res):
    env.set_state(state[:])
    env.set_ready_list(ready_list[:])
    env.set_done_job(done_job[:])
    env.set_tasks(tasks[:])
    env.set_wait_duration(wait_duration[:])
    env.set_cpu_demand(cpu_demand[:])
    env.set_memory_demand(memory_demand[:])
    env.set_tasks_remaing_time(tasks_remaing_time)
    env.set_cpu_res(cpu_res)
    env.set_memory_res(memory_res)
    return 

class TreeNode(object):
    def __init__(self, parent,state,ready_list,done_job,tasks,wait_duration,cpu_demand,memory_demand,tasks_remaing_time,cpu_res,memory_res):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._makespan = 0
        self._state = state
        self._ready_list = ready_list
        self._done_job = done_job
        self._tasks = tasks
        self._wait_duration = wait_duration
        self._cpu_demand = cpu_demand
        self._memory_demand = memory_demand
        self._tasks_remaing_time = tasks_remaing_time
        self._cpu_res = cpu_res
        self._memory_res = memory_res
        self._c = 10
        self._value = 0
        if self._parent != None:
            self.get_value()
    
    def expand(self):
        '''
        扩展树
        '''
        load_current_state(self._state,self._ready_list,self._done_job,self._tasks,self._wait_duration,self._cpu_demand,self._memory_demand,self._tasks_remaing_time,self._cpu_res,self._memory_res)
        available_action = env.return_action_list()
        if available_action:
            for action in available_action:
                load_current_state(self._state,self._ready_list,self._done_job,self._tasks,self._wait_duration,self._cpu_demand,self._memory_demand,self._tasks_remaing_time,self._cpu_res,self._memory_res)
                if action not in self._children:
                    env.step(action)
                    state,ready_list,done_job,tasks,wait_duration,cpu_demand,memory_demand,tasks_remaing_time,cpu_res,memory_res = read_current_state()
                    self._children[action] = TreeNode(self,state,ready_list,done_job,tasks,wait_duration,cpu_demand,memory_demand,tasks_remaing_time,cpu_res,memory_res)
        else:
            print("done")
    
    def get_value(self):
        self._value = self._makespan + self._c * np.sqrt(np.log(self._parent._n_visits+1)/(self._n_visits+1))
        return self._value

    def select(self):
        '''
        在子节中选择具有搜索价值的点
        '''
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value())[1]

    def update(self, makespan):
        # Count visit.
        self._n_visits += 1
        if self._makespan == 0:
            self._makespan = -makespan
        else:
            if -makespan > self._makespan:
                self._makespan = -makespan

    def update_recursive(self, leaf_value):
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None

ppo_agent = PPO(state_dim, action_dim)
print("============================================================================================")
ppo_agent.load(checkpoint_path)
print("Network ID:", run_num_pretrained)
print('PPO agent has been loaded!')

    
initial_state = env.reset()
state,ready_list,done_job,tasks,wait_duration,cpu_demand,memory_demand,tasks_remaing_time,cpu_res,memory_res = read_current_state()

class MCTS(object):
    def __init__(self, state,ready_list,done_job,tasks,wait_duration,cpu_demand,memory_demand,tasks_remaing_time,cpu_res,memory_res,ppo_agent, n_playout=100,):
        self._root = TreeNode(None, state,ready_list,done_job,tasks,wait_duration,cpu_demand,memory_demand,tasks_remaing_time,cpu_res,memory_res)
        self._root.expand() #初始化扩展
        self._n_playout = n_playout
        self._ppo_agent = ppo_agent

    def playout(self):
        i = 0
        while i<200:
            node = self._root
            while i<=100:
                if node.is_leaf():
                    if node._n_visits == 0:
                        cur_state,cur_ready_list,cur_done_job,cur_tasks,cur_wait_duration,cur_cpu_demand,cur_memory_demand,cur_tasks_remaing_time,cur_cpu_res,cur_memory_res = node._state,node._ready_list,node._done_job,node._tasks,node._wait_duration,node._cpu_demand,node._memory_demand,node._tasks_remaing_time,node._cpu_res,node._memory_res
                        makespan = self._roll_out(cur_state,cur_ready_list,cur_done_job,cur_tasks,cur_wait_duration,cur_cpu_demand,cur_memory_demand,cur_tasks_remaing_time,cur_cpu_res,cur_memory_res)
                        node.update_recursive(makespan)
                        break
                    else: 
                        node.expand()
                        node = node.select() 
                else: 
                    node = node.select()        
                i += 1
                print(i)
            
    def _roll_out(self,cur_state,cur_ready_list,cur_done_job,cur_tasks,cur_wait_duration,cur_cpu_demand,cur_memory_demand,cur_tasks_remaing_time,cur_cpu_res,cur_memory_res):
        load_current_state(cur_state,cur_ready_list,cur_done_job,cur_tasks,cur_wait_duration,cur_cpu_demand,cur_memory_demand,cur_tasks_remaing_time,cur_cpu_res,cur_memory_res)
        state = cur_state
        ep_reward = 0
        max_ep_len = 1000  # max timesteps in one episode
        for t in range(1, max_ep_len + 1):
            # select action with policy
            next_state, reward, done, info = self._ppo_agent.select_action(state)
            # saving reward and is_terminals
            ep_reward += reward
            # break; if the episode is over
            state = next_state
            if done:
                makespan = state[0] 
                break
        return makespan  

tree = MCTS(state,ready_list,done_job,tasks,wait_duration,cpu_demand,memory_demand,tasks_remaing_time,cpu_res,memory_res,ppo_agent)
tree.playout()