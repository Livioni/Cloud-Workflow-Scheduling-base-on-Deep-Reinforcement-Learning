import math,torch
from scipy import sparse
import scipy as sp
from gym import spaces
import gym, random
from gym.utils import seeding
from matplotlib import pyplot as plt
from gym.envs.classic_control import rendering
import numpy as np
import networkx as nx
import GCNutils.utils as utils
import GCNutils.models as models


set_dag_size = [20, 30, 40, 50, 60, 70, 80, 90]  # random number of DAG  nodes
set_max_out = [1, 2, 3, 4, 5]  # max out_degree of one node
set_alpha = [0.5, 1.0, 1.5]  # DAG shape
set_beta = [0.0, 0.5, 1.0, 2.0]  # DAG regularity

def DAGs_generate(mode='default', n=10, max_out=2, alpha=1, beta=1.0):
    ##############################################initialize############################################
    if mode != 'default':
        n = random.sample(set_dag_size, 1)[0]
        max_out = random.sample(set_max_out, 1)[0]
        alpha = random.sample(set_alpha, 1)[0]
        beta = random.sample(set_beta, 1)[0]
    else:
        n = 50
        max_out = random.sample(set_max_out, 1)[0]
        alpha = random.sample(set_alpha, 1)[0]
        beta = random.sample(set_beta, 1)[0]

    length = math.floor(math.sqrt(n) / alpha)  # 根据公式计算出来的DAG深度
    mean_value = n / length  # 计算平均长度
    random_num = np.random.normal(loc=mean_value, scale=beta, size=(length, 1))  # 计算每一层的数量，满足正态分布
    ###############################################division#############################################
    position = {'Start': (0, 4), 'Exit': (10, 4)}
    generate_num = 0
    dag_num = 1
    dag_list = []
    for i in range(len(random_num)):  # 对于每一层
        dag_list.append([])
        for j in range(math.ceil(random_num[i])):  # 向上取整
            dag_list[i].append(j)
        generate_num += len(dag_list[i])

    if generate_num != n:  # 不等的话做微调
        if generate_num < n:
            for i in range(n - generate_num):
                index = random.randrange(0, length, 1)
                dag_list[index].append(len(dag_list[index]))
        if generate_num > n:
            i = 0
            while i < generate_num - n:
                index = random.randrange(0, length, 1)  # 随机找一层
                if len(dag_list[index]) < 1:
                    continue
                else:
                    del dag_list[index][-1]
                    i += 1

    dag_list_update = []
    pos = 1
    max_pos = 0
    for i in range(length):
        dag_list_update.append(list(range(dag_num, dag_num + len(dag_list[i]))))
        dag_num += len(dag_list_update[i])
        pos = 1
        for j in dag_list_update[i]:
            position[j] = (3 * (i + 1), pos)
            pos += 5
        max_pos = pos if pos > max_pos else max_pos
        position['Start'] = (0, max_pos / 2)
        position['Exit'] = (3 * (length + 1), max_pos / 2)

    ############################################link#####################################################
    into_degree = [0] * n
    out_degree = [0] * n
    edges = []
    pred = 0

    for i in range(length - 1):
        sample_list = list(range(len(dag_list_update[i + 1])))
        for j in range(len(dag_list_update[i])):
            od = random.randrange(1, max_out + 1, 1)
            od = len(dag_list_update[i + 1]) if len(dag_list_update[i + 1]) < od else od
            bridge = random.sample(sample_list, od)
            for k in bridge:
                edges.append((dag_list_update[i][j], dag_list_update[i + 1][k]))
                into_degree[pred + len(dag_list_update[i]) + k] += 1
                out_degree[pred + j] += 1
        pred += len(dag_list_update[i])

    ######################################create start node and exit node################################
    for node, id in enumerate(into_degree):  # 给所有没有入边的节点添加入口节点作父亲
        if id == 0:
            edges.append(('Start', node + 1))
            into_degree[node] += 1

    for node, od in enumerate(out_degree):  # 给所有没有出边的节点添加出口节点作儿子
        if od == 0:
            edges.append((node + 1, 'Exit'))
            out_degree[node] += 1

    #############################################plot###################################################
    return edges, into_degree, out_degree, position

def plot_DAG(edges, postion):
    plt.figure(1)
    g1 = nx.DiGraph()
    g1.add_edges_from(edges)
    nx.draw_networkx(g1, arrows=True, pos=postion)
    plt.savefig("DAG.png", format="PNG")
    plt.close()
    return plt.clf

def workflows_generator(mode='default', n=10, max_out=2, alpha=1, beta=1.0, t_unit=10, resource_unit=100):
    '''
    随机生成一个DAG任务并随机分配它的持续时间和（CPU，Memory）的需求
    :param mode: DAG按默认参数生成
    :param n: DAG中任务数
    :para max_out: DAG节点最大子节点数
    :return: edges      DAG边信息
             duration   DAG节点持续时间
             demand     DAG节点资源需求数量
    '''
    t = t_unit  # s   time unit
    r = resource_unit  # resource unit
    edges, in_degree, out_degree, position = DAGs_generate(mode, n, max_out, alpha, beta)
    # plot_DAG(edges,position)
    duration = []
    demand = []
    # 初始化持续时间
    for i in range(len(in_degree)):
        if random.random() < 1:
            # duration.append(random.uniform(t,3*t))
            duration.append(random.sample(range(0, 3 * t), 1)[0])
        else:
            # duration.append(random.uniform(5*t,10*t))
            duration.append(random.sample(range(5 * t, 10 * t), 1)[0])
    # 初始化资源需求
    for i in range(len(in_degree)):
        if random.random() < 0.5:
            demand.append((random.uniform(0.25 * r, 0.5 * r), random.uniform(0.05 * r, 0.01 * r)))
        else:
            demand.append((random.uniform(0.05 * r, 0.01 * r), random.uniform(0.25 * r, 0.5 * r)))

    return edges, duration, demand, position

def admatrix(edges,n):
    '''
    返回一个图的邻接矩阵
    :param edges: 生成图边信息
    :param n: 节点个数，不包括'Start'和 'Exit'
    :return adjacency_matrix: 图的邻接矩阵     稀疏形式  
    '''
    graph = nx.DiGraph(edges)
    ndlist = [i for i in range(1,n)]
    adjacency_matrix = nx.to_scipy_sparse_matrix(G = graph,nodelist = ndlist,dtype = np.float32)
    return adjacency_matrix

def convert_to_feature(duration,demand):
    feature = np.array([],dtype=np.float32)
    for line in range(len(duration)):
        feature = np.append(feature,np.array([duration[line],demand[line][0],demand[line][1]],dtype=np.float32))
    feature = sparse.csr_matrix(feature)
    return feature


def gcn_embedding(edges,duration,demand):
    '''
    使用GCN仅编码DAG图信息，每个节点保留三维信息
    :param mode: DAG按默认参数生成
    :param duration: DAG中工作流信息
    :para demand: DAG中工作流信息
    :return: features 节点信息
    :return: adj    邻接矩阵   
    '''
    feature = convert_to_feature(duration,demand)
    features = utils.normalize(feature)
    features = features.toarray().reshape([-1,3])
    features = torch.FloatTensor(features)
    adj = admatrix(edges,len(duration)+1)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = utils.normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse.csr_matrix(adj)
    adj = utils.sparse_mx_to_torch_sparse_tensor(adj)
    return features,adj


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
        self.graph_embedding2_upperbound =  100 * np.ones((1, self.M), dtype=np.float32)
        self.graph_embedding3_upperbound =  100 * np.ones((1, self.M), dtype=np.float32)

        high = np.ravel(np.hstack((
            self.max_time,  # 1 dim
            self.max_cpu_res,  # 1 dim
            self.max_memory_res,  # 1 dim
            self.graph_embedding1_upperbound,
            self.graph_embedding2_upperbound,
            self.graph_embedding3_upperbound
        )))  # totally 38 dim

        low = [0, 0, 0]
        low.extend([-100 for i in range(self.M * 3)])
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
        self.gcn = models.GCN(3,16,3)
        self.gcn.load_state_dict(torch.load('GCN_initialtion/GCN_0.pth', map_location=lambda storage, loc: storage))

        DAGsize = 50
        ##########################################training################################
        print('train datasheet lib.')
        edges_lib_path = '/Users/livion/Documents/GitHub/Cloud-Workflow-Scheduling-base-on-Deep-Reinforcement-Learning/npy/train_datasheet/'+str(DAGsize)+'/edges' + str(DAGsize) +'_lib.npy'
        duration_lib_path = '/Users/livion/Documents/GitHub/Cloud-Workflow-Scheduling-base-on-Deep-Reinforcement-Learning/npy/train_datasheet/'+str(DAGsize)+'/duration' + str(DAGsize) +'_lib.npy'
        demand_lib_path = '/Users/livion/Documents/GitHub/Cloud-Workflow-Scheduling-base-on-Deep-Reinforcement-Learning/npy/train_datasheet/'+str(DAGsize)+'/demand'+str(DAGsize)+'_lib.npy'
        self.edges_lib = np.load(edges_lib_path,allow_pickle=True).tolist()
        self.duration_lib = np.load(duration_lib_path,allow_pickle=True).tolist()
        self.demand_lib = np.load(demand_lib_path,allow_pickle=True).tolist()
        print('load completed.')
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
                self.state = [self.time, self.cpu_res, self.memory_res] + self.graph_embedding1 + self.graph_embedding2 + self.graph_embedding3                                
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

                self.state = [self.time, self.cpu_res, self.memory_res] + self.graph_embedding1 + self.graph_embedding2 + self.graph_embedding3
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
        # self.edges,self.duration,self.demand,self.position = workflows_generator('default')
        self.seed1 = random.randint(0, len(self.duration_lib)-1) 
        self.edges,self.duration,self.demand = self.edges_lib[self.seed1],self.duration_lib[self.seed1],self.demand_lib[self.seed1]
        # self.seed1 += 1
        # if self.seed1 == 100:
        #     self.seed1 = 0
        # print("DAG结构Edges：",self.edges)
        # print("任务占用时间Ti:",self.duration)                    #生成的原始数据
        # print("任务资源占用(res_cpu,res_memory):",self.demand)    #生成的原始数据
        ###初始化一些状态
        #使用GCN编码DAG信息
        features,adj = gcn_embedding(self.edges,self.duration,self.demand)
        output = self.gcn(features,adj)    
        
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
            self.graph_embedding1[ele] = output[ele][0].item()
            self.graph_embedding2[ele] = output[ele][1].item()
            self.graph_embedding3[ele] = output[ele][2].item()

        for i in range(len(self.duration)):
            self.wait_duration_dic[i+1] = self.duration[i]
            self.cpu_demand_dic[i+1] = self.demand[i][0]
            self.memory_demand_dic[i+1] = self.demand[i][1]
        
        self.ready_list = self.update_ready_list(self.ready_list,self.done_job,self.edges[:]) #第一次计算等待任务
        self.state = [self.time, self.cpu_res, self.memory_res] + self.graph_embedding1 + self.graph_embedding2 + self.graph_embedding3
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)

    def render(self, mode="human"):

        return plot_DAG(self.edges, self.position)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None