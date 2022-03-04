import math,argparse
import gym,random
from matplotlib import pyplot as plt
from gym import spaces, logger
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np
import networkx as nx


parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='default', type=str)#parameters setting
parser.add_argument('--n', default=10, type=int)          #number of DAG  nodes
parser.add_argument('--max_out', default=2, type=float)   #max out_degree of one node
parser.add_argument('--alpha',default=1,type=float)       #shape 
parser.add_argument('--beta',default=1.0,type=float)      #regularity
args = parser.parse_args()

set_dag_size = [20,30,40,50,60,70,80,90]             #random number of DAG  nodes       
set_max_out = [1,2,3,4,5]                              #max out_degree of one node
set_alpha = [0.5,1.0,1.5]                            #DAG shape
set_beta = [0.0,0.5,1.0,2.0]                         #DAG regularity

def DAGs_generate(mode = 'default', n = 10, max_out = 2,alpha = 1,beta = 1.0):
    ##############################################initialize############################################
    if mode != 'default':
        args.n = random.sample(set_dag_size,1)[0]
        args.max_out = random.sample(set_max_out,1)[0]
        args.alpha = random.sample(set_alpha,1)[0]
        args.beta = random.sample(set_beta,1)[0]
        args.prob = 0.9
    else: 
        args.n = 30
        args.max_out = random.sample(set_max_out,1)[0]
        args.alpha = random.sample(set_alpha,1)[0]
        args.beta = random.sample(set_beta,1)[0]
        args.prob = 1

    length = math.floor(math.sqrt(args.n)/args.alpha)                                           #根据公式计算出来的DAG深度
    mean_value = args.n/length                                                                  #计算平均长度
    random_num = np.random.normal(loc = mean_value, scale = args.beta,  size = (length,1))      #计算每一层的数量，满足正态分布
    ###############################################division#############################################
    position = {'Start':(0,4),'Exit':(10,4)}
    generate_num = 0
    dag_num = 1
    dag_list = [] 
    for i in range(len(random_num)):                                                            #对于每一层
        dag_list.append([]) 
        for j in range(math.ceil(random_num[i])):                                               #向上取整
            dag_list[i].append(j)
        generate_num += len(dag_list[i])

    if generate_num != args.n:                                                                  #不等的话做微调
        if generate_num < args.n:                                                                 
            for i in range(args.n-generate_num):
                index = random.randrange(0,length,1)
                dag_list[index].append(len(dag_list[index]))
        if generate_num > args.n:
            i = 0
            while i < generate_num-args.n:
                index = random.randrange(0,length,1)                                            #随机找一层
                if len(dag_list[index]) < 1:
                    continue
                else:
                    del dag_list[index][-1]
                    i += 1

    dag_list_update = []
    pos = 1
    max_pos = 0
    for i in range(length):
        dag_list_update.append(list(range(dag_num,dag_num+len(dag_list[i]))))
        dag_num += len(dag_list_update[i])
        pos = 1
        for j in dag_list_update[i]:
            position[j] = (3*(i+1),pos)
            pos += 5
        max_pos = pos if pos > max_pos else max_pos
        position['Start']=(0,max_pos/2)
        position['Exit']=(3*(length+1),max_pos/2)

    ############################################link#####################################################
    into_degree = [0]*args.n            
    out_degree = [0]*args.n             
    edges = []                          
    pred = 0

    for i in range(length-1):
        sample_list = list(range(len(dag_list_update[i+1])))
        for j in range(len(dag_list_update[i])):
            od = random.randrange(1,args.max_out+1,1)
            od = len(dag_list_update[i+1]) if len(dag_list_update[i+1])<od else od
            bridge = random.sample(sample_list,od)
            for k in bridge:
                edges.append((dag_list_update[i][j],dag_list_update[i+1][k]))
                into_degree[pred+len(dag_list_update[i])+k]+=1
                out_degree[pred+j]+=1 
        pred += len(dag_list_update[i])


    ######################################create start node and exit node################################
    for node,id in enumerate(into_degree):#给所有没有入边的节点添加入口节点作父亲
        if id ==0:
            edges.append(('Start',node+1))
            into_degree[node]+=1

    for node,od in enumerate(out_degree):#给所有没有出边的节点添加出口节点作儿子
        if od ==0:
            edges.append((node+1,'Exit'))
            out_degree[node]+=1

    #############################################plot###################################################
    return edges,into_degree,out_degree,position

def plot_DAG(edges,postion):
    plt.figure(1)
    g1 = nx.DiGraph()
    g1.add_edges_from(edges)
    nx.draw_networkx(g1, arrows=True, pos=postion)
    plt.savefig("DAG.png", format="PNG")
    plt.close()
    return plt.clf

def workflows_generator(mode = 'default', n = 10, max_out = 2,alpha = 1,beta = 1.0, t_unit = 10, resource_unit = 100):
    '''
    随机生成一个DAG任务并随机分配它的持续时间和（CPU，Memory）的需求
    :param mode: DAG按默认参数生成
    :param n: DAG中任务数
    :para max_out: DAG节点最大子节点数
    :return: edges      DAG边信息
             duration   DAG节点持续时间
             demand     DAG节点资源需求数量
    '''
    t = t_unit  #s   time unit
    r = resource_unit #resource unit
    edges,in_degree,_,position = DAGs_generate(mode,n,max_out,alpha,beta)
    # plot_DAG(edges,position)
    duration = []
    demand = []
    #初始化持续时间
    for i in range(len(in_degree)):
        if random.random()<args.prob:
            # duration.append(random.uniform(t,3*t))
            duration.append(random.sample(range(1,3*t),1)[0])
        else:
            # duration.append(random.uniform(5*t,10*t))
            duration.append(random.sample(range(5*t,10*t),1)[0])
    #初始化资源需求   
    for i in range(len(in_degree)):
        if random.random()<0.5:
            # demand.append((random.uniform(0.25*r,0.5*r),random.uniform(0.05*r,0.01*r)))
            demand.append((random.uniform(0.25*r,0.5*r),random.uniform(0.05*r,0.01*r)))
        else:
            # demand.append((random.uniform(0.05*r,0.01*r),random.uniform(0.25*r,0.5*r)))
            demand.append((random.uniform(0.05*r,0.01*r),random.uniform(0.25*r,0.5*r)))

    return edges,duration,demand,position



class MyEnv(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        self.M = 10                                                                                      #动作列表长度，不包括-1
        self.t_unit = 10                                                                                 #时间单位，用于生成DAG节点的任务需求时间，80%的概率平均分布于t-3t 20的概率平均分布在10t-15t
        self.cpu_res_unit = 100                                                                          #CPU资源单位，用于生成DAG节点的CPU占用需求，50%的概率的任务为CPU资源需求密集型，随机占用0.25r-0.5r 50%的概率随机占用0.05r-0.01r
        self.memory_res_unit = 100                                                                       #Memory资源单位，用于生成DAG节点的memoory占用需求，50%的概率的任务为memory资源需求密集型，随机占用0.25r-0.5r 50%的概率随机占用0.05r-0.01r
        #以下是各状态的上限定义，好像gym中都需要
        self.max_time = np.array([[999999]],dtype=np.float32)                                            #当前执行时间/当前执行时间上限：没有规定上限，理论上是inf
        self.backlot_max_time = np.array([[999999]],dtype=np.float32)                                    #backlot中任务所需要总时间/backlot中任务所需要总时间上限：没有规定上限，理论上是inf     //超过动作列表长度的任务的信息被算进backlot，backlot没有个数上限
        self.max_cpu_res=  np.array([[self.cpu_res_unit]],dtype=np.float32)                              #计算资源中的CPU总容量
        self.max_memory_res=  np.array([[self.memory_res_unit]],dtype=np.float32)                        #计算资源中的Memory总容量
        self.t_duration = self.t_unit * np.ones((1,self.M),dtype=np.float32)                             #动作列表中每一个任务的执行时间/动作列表中每一个任务的执行时间上限
        self.cpu_res_limt = self.cpu_res_unit* 0.5 * np.ones((1,self.M),dtype=np.float32)                #动作列表中每一个任务CPU资源需求/动作列表中每一个任务CPU资源需求上限
        self.backlot_cpu_res_limt = np.array([[100 * self.cpu_res_unit * 0.5]],dtype=np.float32)         #backlot中任务所需要总CPU资源
        self.memory_res_limt = self.memory_res_unit* 0.5 * np.ones((1,self.M),dtype=np.float32)          #动作列表中每一个任务memory资源需求/动作列表中每一个任务memory资源需求上限
        self.backlot_memory_res_limt = np.array([[100 * self.memory_res_unit * 0.5]],dtype=np.float32)   #backlot中任务所需要总memory资源
        self.max_b_level = np.array([[100]],dtype=np.float32)                                            #b-level上限
        self.max_children = np.array([[100]],dtype=np.float32)                                           #max_children上限
        high = np.ravel(np.hstack((
                                    self.max_time,                  #1 dim
                                    self.max_cpu_res,               #1 dim 
                                    self.max_memory_res,            #1 dim 
                                    self.t_duration,                #10dim
                                    self.cpu_res_limt,              #10dim
                                    self.memory_res_limt,           #10dim
                                    self.max_b_level,               #1 dim
                                    self.max_children,              #1 dim 
                                    self.backlot_max_time,          #1 dim
                                    self.backlot_cpu_res_limt,      #1 dim
                                    self.backlot_memory_res_limt,   #1 dim
                                    )))                             # totally 38 dim
        low = np.array([0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0],dtype=np.float32)
        self.action_space = spaces.Discrete(11)#{-1,0,1,2,3,4,5,6,7,8,9}
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        ##状态信息
        self.time = 0                                           #整体环境时钟，DAG执行已花费时间
        self.cpu_res = 100                                      #当前计算资源的CPU容量
        self.memory_res = 100                                   #当前计算资源的memory容量
        self.b_level = None                                     #b-level值  
        self.children_num = None                                #子节点数
        self.edges = []                                         #随机生成DAG的信息
        self.available_list = [-1] * self.M                     #调度输入个数 尽量大于readylist 取self.M，不足补-1   
        self.duration = []                                      #随机生成DAG的时间占用信息
        self.demand = []                                        #backlot的总资源占用信息
        self.wait_duration = [-1] * self.M                      #随机生成的DAG时间占用信息
        self.cpu_demand = [-1] * self.M                         #随机生成的DAG cpu 资源占用信息
        self.memory_demand = [-1] * self.M                      #随机生成的DAG memory 资源占用信息
        self.ready_list = []                                    #满足依赖关系的DAG,可能被执行，但不满足               
        self.done_job = []                                      #已完成的任务ID
        self.position = []                                      #当前DAG画图坐标
        self.backlot_time = 0                                   #backlot的总时间占用信息
        self.backlot_cpu_res = 0                                #backlot的总CPU占用信息 
        self.backlot_memory_res = 0                             #backlot的总memory占用信息 
        ##状态转移信息和中间变量
        self.tasks = []                                         #计算资源上挂起的任务
        self.tasks_remaing_time = {}                            #计算资源上挂起的任务剩余执行时间
        self.DeepRM_reward = 0
        self.seed()
        self.seed1 = 0
        self.viewer = None
        self.state = None
        self.edges_lib = []
        self.duration_lib = []
        self.demand_lib = []
        
        DAGsize = 40
        ##########################################training################################
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

    def search_for_predecessor(self,node,edges):
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

    def search_for_successors(self,node,edges):
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

    def update_ready_list(self,ready_list,done_job,edges):
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
                all_succ.extend(self.search_for_successors(ready_list[i],edges))
        else:
            ready_list = ['Start']
            return self.update_ready_list(ready_list,['Start'],edges)
        for i in all_succ:
            preds = self.search_for_predecessor(i,edges)
            if (set(done_job)>= set(preds)):
                ready_list.append(i)
        for job in done_job : 
            if job in ready_list: ready_list.remove(job)
        return sorted(set(ready_list), key = ready_list.index)

    def find_b_level(self,edges,done_job):
        '''
        计算当前未完成DAG的b-level（lenth of the longest path) 不包括 Start 和 Exit节点
        :para edges: DAG边信息(注意最好传列表的值（edges[:]）进去而不是传列表的地址（edges）!!!)
        :param done_job: DAG中已完成的任务id列表 注意done_job是所有任务id的子集
        :return: b_level
        '''
        for i in range(len(done_job)):
            preds = self.search_for_predecessor(done_job[i],edges)
            for j in preds: edges.pop(edges.index((j,done_job[i])))
        g1 = nx.DiGraph()
        g1.add_edges_from(edges)
        b_level_road = nx.dag_longest_path(g1)
        return len(b_level_road[1:-1])

    def find_children_num(self,ready_list,edges):
        '''
        计算ready task的子节点数，作为状态的输入
        :param ready_list: DAG中准备好的任务列表
        :para edges: DAG边信息(注意最好传列表的值（edges[:]）进去而不是传列表的地址（edges）!!!)
        :return: 子节点的数量，不包括'Exit'
        '''
        children = []
        for job in ready_list:
            children.extend(self.search_for_successors(job,edges))
        length = 0
        while (length != len(children)):
            length = len(children)
            for job in children:
                if job != 'Exit':
                    children.extend(self.search_for_successors(job,edges))
                children = sorted(set(children), key = children.index)
        return len(set(children))-1    

    def res_is_available(self,action):
        '''
        判断当前选择的动作是否能在计算资源上执行
        :para action {-1,0,1,2,3,4,5,6,7,8,9}
        :return: True 可以被执行 False 不能被执行
        '''
        task_cpu_demand = self.cpu_demand[action]
        task_memory_demand = self.memory_demand[action]
        if self.wait_duration[action] != -1.0:
            return True if (task_cpu_demand <= self.cpu_res) & (task_memory_demand <= self.memory_res) else False #判断资源是否满足要求
        else: 
            return False


    def check_action(self,action):
        '''
        判断当前选择的动作是否有效
        :para action {-1,0,1,2,3,4,5,6,7,8,9}
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

    def check_episode_finish(self):
        '''
        判断当前幕是否已经执行完成
        :para None
        :return: True 完成一幕了 False 还未完成一幕
        '''
        if len(self.done_job) == len(self.duration):
            return True   
        else: False


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def pend_task(self,action):
        job_id = self.ready_list[action]
        self.tasks.append(job_id)#self.ready_list[action]表示的是任务ID 
        self.tasks_remaing_time[job_id] = self.wait_duration_dic[job_id] 
        self.cpu_res -= self.cpu_demand_dic[job_id]                                                                #当前CPU资源容量-这个任务需求的CPU资源
        self.memory_res -= self.memory_demand_dic[job_id]
        self.wait_duration[action] = -1.0                                                                       #waiting列表中挂起的任务信息变为-1
        self.cpu_demand[action] = -1.0                                                                          #waiting列表中挂起的任务信息变为-1
        self.memory_demand[action] = -1.0 


    def step(self, action):
        '''
        状态转移过程
        :para action {-1,0,1,2,3,4,5,6,7,8,9}
        :return: 下一个状态，回报，是否完成，调试信息
                 False ，要求重新采样动作
        ''' 
        '''
        self.ready_list是当前可执行的任务，包括了在机器上挂起的
        '''
        done = self.check_episode_finish()
        if done:
            # print("全部任务已完成")
            return np.array(self.state, dtype=np.float32), 0, done, [True, self.tasks]

        if (action >= 0 & action<=9): 
            if (self.check_action(action)):
                self.pend_task(action)
                self.state = [self.time, self.cpu_res, self.memory_res] + self.wait_duration + self.cpu_demand + \
                self.memory_demand + [self.b_level, self.children_num, self.backlot_time, self.backlot_cpu_res, self.backlot_memory_res]                                     
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
                self.wait_duration = [-1] * self.M                                                                      
                self.cpu_demand = [-1] * self.M                                                                         
                self.memory_demand = [-1] * self.M  
                self.backlot_time = 0                                                                                  
                self.backlot_cpu_res = 0                                                                                
                self.backlot_memory_res = 0 
                for i in range(len(self.ready_list)):
                    job_id = self.ready_list[i]
                    if self.ready_list[i] in self.tasks:
                        continue                                   
                    if i < self.M: 
                        self.wait_duration[i] = self.wait_duration_dic[job_id]                                       
                        self.cpu_demand[i] = self.cpu_demand_dic[job_id]
                        self.memory_demand[i] = self.memory_demand_dic[job_id]
                    else:
                        self.backlot_time += self.wait_duration_dic[job_id]
                        self.backlot_cpu_res += self.cpu_demand_dic[job_id]
                        self.backlot_memory_res += self.memory_demand_dic[job_id]
                self.b_level = self.find_b_level(self.edges[:],self.done_job)
                self.children_num = self.find_children_num(self.ready_list,self.edges[:]) 
                self.state = [self.time, self.cpu_res, self.memory_res] + self.wait_duration + self.cpu_demand + \
                        self.memory_demand + [self.b_level, self.children_num, self.backlot_time, self.backlot_cpu_res, self.backlot_memory_res]
                return np.array(self.state, dtype=np.float32), reward, done, [True, self.tasks]
            else:
                return np.array(self.state, dtype=np.float32), 0, 0, [False, self.tasks]
                
    def save_50dag(self):
            edges = [(1, 3), (2, 3), (3, 5), (4, 11), (4, 13), (5, 9), (6, 8), (7, 16), (8, 16), (8, 15), (9, 16), (9, 15), (10, 16), (10, 15), (11, 14), (12, 15), (13, 16), (17, 24), (18, 24), (19, 24), (19, 23), (20, 22), (20, 24), (21, 23), (21, 24), (22, 25), (23, 29), (23, 31), (24, 30), (25, 33), (26, 32), (26, 33), (27, 33), (28, 33), (29, 32), (29, 33), (30, 32), (30, 33), (31, 33), (31, 32), (32, 34), (32, 35), (33, 34), (34, 41), (35, 41), (36, 41), (37, 41), (38, 41), (39, 41), (40, 41), (41, 42), (42, 47), (43, 50), (44, 46), ('Start', 1), ('Start', 2), ('Start', 4), ('Start', 6), ('Start', 7), ('Start', 10), ('Start', 12), ('Start', 17), ('Start', 18), ('Start', 19), ('Start', 20), ('Start', 21), ('Start', 26), ('Start', 27), ('Start', 28), ('Start', 36), ('Start', 37), ('Start', 38), ('Start', 39), ('Start', 40), ('Start', 43), ('Start', 44), ('Start', 45), ('Start', 48), ('Start', 49), (14, 'Exit'), (15, 'Exit'), (16, 'Exit'), (45, 'Exit'), (46, 'Exit'), (47, 'Exit'), (48, 'Exit'), (49, 'Exit'), (50, 'Exit')]       
            duration = [20.358396308961733, 22.067907217911092, 17.754237679091347, 20.615827903960874, 11.108342471522475, 28.360987765703182, 27.027983215737336, 12.510292895518278, 15.08195718692922, 18.04858291907044, 10.465057748361426, 21.978882344989835, 19.19599492027384, 10.249260422546458, 27.49138941453423, 19.80640773179836, 29.29675544705301, 27.35020534279648, 18.84192558794532, 20.143369578761778, 12.190528815882598, 13.234386695259097, 12.307559634581738, 21.903343578174272, 28.316744012207167, 20.076252358675184, 27.799552388706942, 12.430029583264833, 15.21769102989123, 27.147320389454816, 28.84881423495435, 10.289599820894335, 12.632821110128438, 18.573024740625865, 18.43744696585427, 13.315260733366918, 27.319530652728005, 15.723256567039435, 13.058172187712394, 16.11771067917175, 26.783813986688145, 28.281370045457514, 25.84972440753974, 16.37206986195357, 11.53940573303627, 25.862079221725516, 12.421899016945071, 16.974721130621607, 27.899423208626867, 23.94427001090075]
            demand = [(1.3327185569866247, 26.578004698672753), (2.1451046250725723, 34.96995311494437), (38.96538519195313, 1.214281117134822), (41.32430130707104, 2.2386522787974825), (28.161635048523543, 4.392188187795879), (1.4919583053683931, 46.131744883214324), (37.694456781667846, 3.8133279239502422), (3.6368049325377436, 34.71688811267414), (2.7556326104892874, 44.471943054415604), (41.73639848335215, 4.936652730701006), (4.436997442877024, 31.465636395464944), (2.814360099685854, 40.446334042202174), (35.186813645741445, 2.247151441306695), (25.82434507252789, 2.9766864288454853), (27.101238160713798, 1.144648269362878), (28.924007958218144, 4.76502164770424), (4.885493221424168, 35.384058224086274), (28.280955350534043, 2.3949557238027506), (1.788647831259305, 29.007483373421877), (27.58762314814667, 4.8220637886542255), (1.977926680832649, 29.989526921497045), (4.055644120263752, 36.226272991299524), (48.42426904935452, 4.613942532865045), (4.310367250302273, 45.19265878822756), (2.8565533228916937, 32.80694794647271), (45.549515155378316, 2.220203393513902), (36.60799821838276, 2.9771357093982647), (26.645801312631402, 3.275713761622243), (49.65665844011782, 3.8342616067942306), (3.343684884036226, 47.46146644975276), (2.253510924917405, 49.58977962846717), (3.6980672529136167, 48.45704294470164), (2.405504803186821, 38.727302500903534), (1.4910071921243708, 41.520388957677326), (38.07283445347876, 1.8164910473630762), (29.353649374593076, 4.229702278616407), (2.3865861867323592, 41.22121108097354), (47.83631588351712, 1.7182419550989074), (4.128567987723979, 34.03809646761347), (33.55576743467706, 1.0107826458205564), (1.9049057975768569, 37.0202698325086), (1.0708510581687496, 34.81946275349077), (2.1341585691351006, 44.346433430191766), (35.17389850481341, 3.300765362443588), (4.39564619088428, 38.17842642610517), (1.086674737013217, 37.94751503748752), (3.4281272781378886, 41.03412800662693), (26.273228934920912, 3.2383484587780687), (46.612838068039466, 1.8298177205408117), (3.750948259062607, 37.78093008235358)]
            return edges,duration,demand

    def reset(self):
        '''
        初始化状态
        :para None
        :return: 初始状态，每一次reset就是重新一幕
        '''
        # self.seed1 = random.randint(0, 99) 
        self.edges,self.duration,self.demand = self.edges_lib[self.seed1],self.duration_lib[self.seed1],self.demand_lib[self.seed1]
        self.seed1 += 1
        if self.seed1 == 100:
            self.seed1 = 0
        ###随机生成一个workflow
        # self.edges,self.duration,self.demand,self.position = workflows_generator('default')
        # self.edges,self.duration,self.demand = self.save_10dag()
        # print("DAG结构Edges：",self.edges)
        # print("任务占用时间Ti:",self.duration)                    #生成的原始数据
        # print("任务资源占用(res_cpu,res_memory):",self.demand)    #生成的原始数据
        ###初始化一些状态
        self.time,self.cpu_res,self.memory_res = 0,100,100      
        self.done_job = []
        self.ready_list = []     
        self.wait_duration_dic = {}                             #随机生成的DAG时间占用信息
        self.cpu_demand_dic = {}                                #随机生成的DAG CPU资源占用信息
        self.memory_demand_dic = {}                             #随机生成的DAG Memo资源占用信息
        self.backlot_time = 0                                   #backlot的总时间占用信息
        self.backlot_cpu_res = 0                                #backlot的总CPU占用信息 
        self.backlot_memory_res = 0                             #backlot的总memory占用信息 
        self.tasks_remaing_time = {}
        self.tasks = []
        self.DeepRM_reward = 0
        self.ready_list = self.update_ready_list(self.ready_list,self.done_job,self.edges[:])
        # print("ready list:",self.ready_list)
        for i in range(len(self.duration)):
            self.wait_duration_dic[i+1] = self.duration[i]
            self.cpu_demand_dic[i+1] = self.demand[i][0]
            self.memory_demand_dic[i+1] = self.demand[i][1]
        # print("wait_duration_dic:",self.wait_duration_dic)
        # print("cpu_demand_dic:", self.cpu_demand_dic)
        # print("memory_demand_dic:",self.memory_demand_dic)
        for i in range(len(self.ready_list)):
            job_id = self.ready_list[i]
            if i < self.M: 
                self.wait_duration[i] = self.wait_duration_dic[job_id]
                self.cpu_demand[i] = self.cpu_demand_dic[job_id]
                self.memory_demand[i] = self.memory_demand_dic[job_id]
            else:
                self.backlot_time += self.wait_duration_dic[job_id]
                self.backlot_cpu_res += self.cpu_demand_dic[job_id]
                self.backlot_memory_res += self.memory_demand_dic[job_id]

        self.average_time_duration = 0
        for i in self.wait_duration_dic.values():
            self.average_time_duration += i
        self.average_time_duration /= args.n 

        self.b_level = self.find_b_level(self.edges[:],self.done_job) 
        self.children_num = self.find_children_num(self.ready_list,self.edges[:])

        self.state = [self.time,self.cpu_res,self.memory_res] + self.wait_duration + self.cpu_demand + \
            self.memory_demand + [self.b_level, self.children_num, self.backlot_time, self.backlot_cpu_res, self.backlot_memory_res]
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)

    def render(self, mode="human"):

        return plot_DAG(self.edges,self.position)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

