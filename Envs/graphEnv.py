import torch
from gym import spaces
import gym
from gym.utils import seeding
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import GCNutils.models as models
from . import utils 

class DecimaGNN(nn.Module):  # 策略网络
    def __init__(self, input_size, output_size):
        super(DecimaGNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear1 = nn.Linear(self.input_size, 16)
        self.linear2 = nn.Linear(16, self.output_size)

    def forward(self, input_feature):
        output = self.linear1(input_feature)
        output = F.leaky_relu(self.linear2(output))
        return output  # 输出动作概率分布

class graphEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        self.M = 50  # 动作列表长度
        self.t_unit = 10  # 时间单位，用于生成DAG节点的任务需求时间，80%的概率平均分布于t-3t 20的概率平均分布在10t-15t
        self.cpu_res_unit = 100  # CPU资源单位，用于生成DAG节点的CPU占用需求，50%的概率的任务为CPU资源需求密集型，随机占用0.25r-0.5r 50%的概率随机占用0.05r-0.01r
        self.memory_res_unit = 100  # Memory资源单位，用于生成DAG节点的memoory占用需求，50%的概率的任务为memory资源需求密集型，随机占用0.25r-0.5r 50%的概率随机占用0.05r-0.01r

        # 以下是各状态的上限定义，好像gym中都需要
        self.max_time = np.array([[999999]], dtype=np.float32)  # 当前执行时间/当前执行时间上限：没有规定上限
        self.max_cpu_res = np.array([[self.cpu_res_unit]], dtype=np.float32)  # 计算资源中的CPU总容量
        self.max_memory_res = np.array([[self.memory_res_unit]], dtype=np.float32)  # 计算资源中的Memory总容量

        self.graph_embedding1_upperbound =  100 * np.ones((1, self.M), dtype=np.float32)
        # self.graph_embedding2_upperbound =  100 * np.ones((1, self.M), dtype=np.float32)
        # self.graph_embedding3_upperbound =  100 * np.ones((1, self.M), dtype=np.float32)

        high = np.ravel(np.hstack((
            self.max_time,  # 1 dim
            self.max_cpu_res,  # 1 dim
            self.max_memory_res,  # 1 dim
            self.graph_embedding1_upperbound
            # self.graph_embedding2_upperbound,
            # self.graph_embedding3_upperbound
        )))  # totally 38 dim

        low = [0, 0, 0]
        low.extend([-100 for i in range(self.M * 1)])
        low = np.array(low, dtype=np.float32)
        self.action_space = spaces.Discrete(self.M+1)  # {-1,0,1,2,3,4,5,6,7,8,9}
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        ##状态信息
        self.time = 0  # 整体环境时钟，DAG执行已花费时间
        self.cpu_res = 100  # 当前计算资源的CPU容量
        self.memory_res = 100  # 当前计算资源的memory容量
        self.graph_embedding1 = [-1] * self.M
        self.graph_embedding2 = [-1] * self.M
        self.graph_embedding3 = [-1] * self.M

        self.edges = []  # 随机生成DAG的信息
        self.ready_list = []  # 满足依赖关系的DAG,可能被执行，但不满足
        self.done_job = []  # 已完成的任务ID
        ##状态转移信息和中间变量
        self.tasks = []  # 计算资源上挂起的任务
        self.tasks_remaing_time = {}  # 计算资源上挂起的任务剩余执行时间
        self.seed()
        self.seed1 = 0
        self.viewer = None
        self.state = None

        ##########################################GCN embeddings################################
        self.gcn = models.GCN(3,16,1)
        self.gcn.load_state_dict(torch.load('GCN_initialization/GCN_1.pth', map_location=lambda storage, loc: storage))
        print('Gcn parameters have been loaded.')
        # self.NonLinearNw1 = DecimaGNN(3,3)
        self.NonLinearNw2 = DecimaGNN(1,1)
        self.NonLinearNw3 = DecimaGNN(1,1)
        # self.NonLinearNw1.load_state_dict(torch.load('GCN_initialization/NonLinearNw1.pth', map_location=lambda storage, loc: storage))
        self.NonLinearNw2.load_state_dict(torch.load('GCN_initialization/NonLinearNw2.pth', map_location=lambda storage, loc: storage))
        self.NonLinearNw3.load_state_dict(torch.load('GCN_initialization/NonLinearNw3.pth', map_location=lambda storage, loc: storage))

        DAGsize = 30
        ##########################################training#####################################
        # print('train datasheet lib.')
        # edges_lib_path = '/Users/livion/Documents/GitHub/Cloud-Workflow-Scheduling-base-on-Deep-Reinforcement-Learning/npy/train_datasheet/'+str(DAGsize)+'/edges' + str(DAGsize) +'_lib.npy'
        # duration_lib_path = '/Users/livion/Documents/GitHub/Cloud-Workflow-Scheduling-base-on-Deep-Reinforcement-Learning/npy/train_datasheet/'+str(DAGsize)+'/duration' + str(DAGsize) +'_lib.npy'
        # demand_lib_path = '/Users/livion/Documents/GitHub/Cloud-Workflow-Scheduling-base-on-Deep-Reinforcement-Learning/npy/train_datasheet/'+str(DAGsize)+'/demand'+str(DAGsize)+'_lib.npy'
        # self.edges_lib = np.load(edges_lib_path,allow_pickle=True).tolist()
        # self.duration_lib = np.load(duration_lib_path,allow_pickle=True).tolist()
        # self.demand_lib = np.load(demand_lib_path,allow_pickle=True).tolist()
        # print('load completed.')
        ##########################################testing################################
        # print('test datasheet loaded.')
        # edges_lib_path = '/Users/livion/Documents/GitHub/Cloud-Workflow-Scheduling-base-on-Deep-Reinforcement-Learning/npy/test_datasheet/'+str(DAGsize)+'/edges' + str(DAGsize) +'_lib.npy'
        # duration_lib_path = '/Users/livion/Documents/GitHub/Cloud-Workflow-Scheduling-base-on-Deep-Reinforcement-Learning/npy/test_datasheet/'+str(DAGsize)+'/duration' + str(DAGsize) +'_lib.npy'
        # demand_lib_path = '/Users/livion/Documents/GitHub/Cloud-Workflow-Scheduling-base-on-Deep-Reinforcement-Learning/npy/test_datasheet/'+str(DAGsize)+'/demand'+str(DAGsize)+'_lib.npy'
        # self.edges_lib = np.load(edges_lib_path,allow_pickle=True).tolist()
        # self.duration_lib = np.load(duration_lib_path,allow_pickle=True).tolist()
        # self.demand_lib = np.load(demand_lib_path,allow_pickle=True).tolist()
        # print('load completed.')

    def search_for_predecessor(self, node, edges):
        '''
        寻找前继节点
        :param node: 需要查找的节点id
        :param edges: DAG边信息
        :return: node的前继节点id列表
        '''
        map = {}
        if node == 'Start': return print("error, 'Start' node do not have predecessor!")
        for i in range(len(edges)):
            if edges[i][1] in map.keys():
                map[edges[i][1]].append(edges[i][0])
            else:
                map[edges[i][1]] = [edges[i][0]]
        succ = map[node]
        return succ

    def search_for_successors(self, node, edges):
        '''
        寻找后续节点
        :param node: 需要查找的节点id
        :param edges: DAG边信息(注意最好传列表的值（edges[:]）进去而不是传列表的地址（edges）!!!)
        :return: node的后续节点id列表
        '''
        map = {}
        if node == 'Exit': return print("error, 'Exit' node do not have successors!")
        for i in range(len(edges)):
            if edges[i][0] in map.keys():
                map[edges[i][0]].append(edges[i][1])
            else:
                map[edges[i][0]] = [edges[i][1]]
        pred = map[node]
        return pred

    def search_for_all_successors(self,node, edges):
        save = node
        node = [node]
        for ele in node:
            succ = self.search_for_successors(ele,edges)
            if(len(succ)==1 and succ[0]=='Exit'):
                break
            for item in succ:
                if item in node:
                    continue
                else:
                    node.append(item)
        node.remove(save)
        return node

    def update_ready_list(self, ready_list, done_job, edges):
        '''
        根据已完成的任务更新当前可以执行的task列表，满足DAG的依赖关系。并不表明它可以被执行，因为还要受资源使用情况限制
        :param ready_list: 上一时刻可用task列表
        :param done_job: DAG中已完成的任务id列表
        :para edges: DAG边信息(注意最好传列表的值（edges[:]）进去而不是传列表的地址（edges）!!!)
        :return: 更新完成后的task列表
        '''
        all_succ = []
        preds = []
        if ready_list:
            for i in range(len(ready_list)):
                all_succ.extend(self.search_for_successors(ready_list[i], edges))
        else:
            ready_list = ['Start']
            return self.update_ready_list(ready_list, ['Start'], edges)
        for i in all_succ:
            preds = self.search_for_predecessor(i, edges)
            if (set(done_job) >= set(preds)):
                ready_list.append(i)
        for job in done_job:
            if job in ready_list: ready_list.remove(job)
        return sorted(set(ready_list), key=ready_list.index)

    def check_episode_finish(self):
        '''
        判断当前幕是否已经执行完成
        :para None
        :return: True 完成一幕了 False 还未完成一幕
        '''
        if len(self.done_job) == len(self.duration):
            return True
        else:
            False

    def res_is_available(self,action):
        '''
        判断当前选择的动作是否能在计算资源上执行
        :para action {-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29}
        :return: True 可以被执行 False 不能被执行
        '''
        job_id = self.ready_list[action]
        task_cpu_demand = self.cpu_demand_dic[job_id]
        task_memory_demand = self.memory_demand_dic[job_id]
        if self.graph_embedding1[job_id-1] != -1.0:
            return True if (task_cpu_demand <= self.cpu_res) & (task_memory_demand <= self.memory_res) else False #判断资源是否满足要求
        else: 
            return False

    def check_action(self,action):
        '''
        判断当前选择的动作是否有效
        :para action {-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29}
        :return: True 有效 False 无效
        '''
        if(action < len(self.ready_list)):
            if(self.ready_list[action] not in self.done_job):
                if(self.ready_list[action] not in self.tasks):
                    return self.res_is_available(action)
                else:
                    return False 
            else:
                return False    
        else:
            return False

    def pend_task(self,action):
        job_id = self.ready_list[action]
        self.tasks.append(job_id)#self.ready_list[action]表示的是任务ID 
        self.tasks_remaing_time[job_id] = self.wait_duration_dic[job_id] 
        self.cpu_res -= self.cpu_demand_dic[job_id]                                                                #当前CPU资源容量-这个任务需求的CPU资源
        self.memory_res -= self.memory_demand_dic[job_id]
        self.graph_embedding1[job_id-1] = -1.0
        self.graph_embedding2[job_id-1] = -1.0
        self.graph_embedding3[job_id-1] = -1.0

    def Decima_encoder(self,edges,duration,demand):
        '''
        使用Decima编码器编码DAG图信息
        :param duration: 工作流信息
        :param edges: DAG边信息
        :param demand: 工作流信息
        :return: embeddings dag图的节点编码信息，以字典形式储存。
        '''
        raw_embeddings = [] #原始节点feature
        embeddings =  {}  #编码后的feature字典  job_id : embedding

        cpu_demands = [demand[i][0] for i in range(len(demand))]
        memory_demands = [demand[i][1] for i in range(len(demand))]
        for exetime,cpu_demand,memory_demand in zip(duration,cpu_demands,memory_demands):
            raw_embeddings.append([exetime,cpu_demand,memory_demand])
        raw_embeddings = np.array(raw_embeddings,dtype=np.float32)
        raw_embeddings = torch.from_numpy(raw_embeddings)
        features,adj = utils.gcn_embedding(self.edges,self.duration,self.demand)
        embeddings1 = self.gcn(features,adj)  #第一层初始编码信息
        # embeddings1 = self.gcn(raw_embeddings) 

        pred0 = self.search_for_predecessor('Exit',edges[:])
        for ele in pred0:
            embeddings[ele] = embeddings1[ele-1].data

        while(len(embeddings.keys())<len(duration)):
            box = []
            for ele in pred0:
                pred = self.search_for_predecessor(ele,edges[:])
                for i in pred:
                    if i in embeddings.keys():
                        continue
                    if i == 'Start':
                        continue
                    succ = self.search_for_all_successors(i,edges[:])
                    g = torch.tensor([0],dtype=torch.float32)
                    for j in succ:
                        g +=self.NonLinearNw2(embeddings[j])
                    embeddings[i] = self.NonLinearNw3(g) + embeddings1[i]
                    box.append(i)
            pred0 = box
        return embeddings

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        '''
        状态转移过程
        :para action {-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29}
        :return: 下一个状态，回报，是否完成，调试信息
                 False ，要求重新采样动作
        ''' 
        '''
        self.ready_list是当前可执行的任务，包括了在机器上挂起的
        '''
        done = self.check_episode_finish()
        if done:
            return np.array(self.state, dtype=np.float32), 0, done, [True, self.tasks]

        if (action >= 0 & action<=self.M-1): 
            if (self.check_action(action)):
                self.pend_task(action)
                self.state = [self.time, self.cpu_res, self.memory_res] + self.graph_embedding1 #+ self.graph_embedding2 + self.graph_embedding3                                
                reward = 0.0                                                                                         #时间步没动，收益为0
                done = self.check_episode_finish()
                return np.array(self.state, dtype=np.float32), reward, done, [True, self.tasks]
            else:
                return np.array(self.state, dtype=np.float32), 0, 0, [False, self.tasks] 
        else:
            if self.tasks:
                self.tasks_remaing_time_list = sorted(self.tasks_remaing_time.items(),key = lambda x:x[1])  #排序当前挂起任务的执行时间
                job_id = self.tasks_remaing_time_list[0][0]                                                 
                time_shift = self.tasks_remaing_time_list[0][1]                                             #记录最小执行时间的长度
                self.time += time_shift                                                                     #改变时间戳
                self.done_job.append(job_id)                                                                #更新已完成的任务
                done = self.check_episode_finish()
                if done:
                    return np.array(self.state, dtype=np.float32), 0, done, [True, self.tasks]
                self.cpu_res += self.cpu_demand_dic[job_id]                                                 #释放CPU资源
                self.memory_res += self.memory_demand_dic[job_id]                                           #释放Memory资源
                self.tasks.remove(job_id)                                                                   #删除pending的task
                del self.tasks_remaing_time[job_id]                                                         #删除任务剩余时间信息
                for i in self.tasks_remaing_time.keys():
                    self.tasks_remaing_time[i] -= time_shift

                reward = -time_shift/self.t_unit

                self.ready_list = self.update_ready_list(self.ready_list,self.done_job,self.edges[:])       #更新ready_list

                self.state = [self.time, self.cpu_res, self.memory_res] + self.graph_embedding1 #+ self.graph_embedding2 + self.graph_embedding3
                return np.array(self.state, dtype=np.float32), reward, done, [True, self.tasks]
            else:
                return np.array(self.state, dtype=np.float32), 0, 0, [False, self.tasks]
        
    def reset(self):
        '''
        初始化状态
        :para None
        :return: 初始状态，每一次reset就是重新一幕
        '''
        ###随机生成一个workflow
        self.edges,self.duration,self.demand,self.position = utils.workflows_generator('default',n=30)
        # self.seed1 = random.randint(0, len(self.duration_lib)-1) 
        # self.edges,self.duration,self.demand = self.edges_lib[self.seed1],self.duration_lib[self.seed1],self.demand_lib[self.seed1]
        # self.seed1 += 1
        # if self.seed1 == 1000:
        #     self.seed1 = 0
        # print("DAG结构Edges：",self.edges)
        # print("任务占用时间Ti:",self.duration)                    #生成的原始数据
        # print("任务资源占用(res_cpu,res_memory):",self.demand)    #生成的原始数据
        ###初始化一些状态
        #使用GCN编码DAG信息  
        output = self.Decima_encoder(edges=self.edges,duration=self.duration,demand=self.demand)
        
        ####初始化变量
        self.time,self.cpu_res,self.memory_res = 0,100,100
        self.ready_list = []  # 满足依赖关系的DAG,可能被执行，但不满足
        self.done_job = []  # 已完成的任务ID
        self.tasks = []  # 计算资源上挂起的任务
        self.tasks_remaing_time = {}  # 计算资源上挂起的任务剩余执行时间
        self.ready_list = []
        self.wait_duration_dic = {}                             #随机生成的DAG时间占用信息 job_id : value
        self.cpu_demand_dic = {}                                #随机生成的DAG CPU资源占用信息 job_id : value
        self.memory_demand_dic = {}                             #随机生成的DAG Memo资源占用信息 job_id : value
        for ele in range(len(self.duration)):
            self.graph_embedding1[ele] = output[ele+1][0].item()
            # self.graph_embedding2[ele] = output[ele+1][1].item()
            # self.graph_embedding3[ele] = output[ele+1][2].item()

        for i in range(len(self.duration)):
            self.wait_duration_dic[i+1] = self.duration[i]
            self.cpu_demand_dic[i+1] = self.demand[i][0]
            self.memory_demand_dic[i+1] = self.demand[i][1]
        
        self.ready_list = self.update_ready_list(self.ready_list,self.done_job,self.edges[:]) #第一次计算等待任务
        self.state = [self.time, self.cpu_res, self.memory_res] + self.graph_embedding1 # + self.graph_embedding2 + self.graph_embedding3
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)

    def render(self, mode="human"):
        return plot_DAG(self.edges, self.position)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None