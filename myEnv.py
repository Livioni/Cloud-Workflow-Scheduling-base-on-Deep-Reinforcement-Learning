import math,argparse
import gym,random
from matplotlib import pyplot as plt
from gym import spaces, logger
from gym.utils import seeding
from networkx.classes.function import edges
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
        args.beta = random.sample(set_alpha,1)[0]
    else: 
        args.n = n
        args.max_out = max_out
        args.alpha = alpha
        args.beta = beta

    length = math.floor(math.sqrt(args.n)/args.alpha)
    mean_value = args.n/length
    random_num = np.random.normal(loc = mean_value, scale = args.beta,  size = (length,1))    
    ###############################################division#############################################
    position = {'Start':(0,4),'Exit':(10,4)}
    generate_num = 0
    dag_num = 1
    dag_list = [] 
    for i in range(len(random_num)):
        dag_list.append([]) 
        for j in range(math.ceil(random_num[i])):
            dag_list[i].append(j)
        generate_num += math.ceil(random_num[i])

    if generate_num != args.n:
        if generate_num<args.n:
            for i in range(args.n-generate_num):
                index = random.randrange(0,length,1)
                dag_list[index].append(len(dag_list[index]))
        if generate_num>args.n:
            i = 0
            while i < generate_num-args.n:
                index = random.randrange(0,length,1)
                if len(dag_list[index])==1:
                    i = i-1 if i!=0 else 0
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
    g1 = nx.DiGraph()
    g1.add_edges_from(edges)
    nx.draw_networkx(g1, arrows=True, pos=postion)
    plt.savefig("DAG.png", format="PNG")
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
    edges,in_degree,out_degree,position = DAGs_generate(mode,n,max_out,alpha,beta)
    plot_DAG(edges,position)
    duration = []
    demand = []
    #初始化持续时间
    for i in range(len(in_degree)):
        if random.random()<0.8:
            duration.append(random.uniform(t,3*t))
        else:
            duration.append(random.uniform(10*t,15*t))
    #初始化资源需求   
    for i in range(len(in_degree)):
        if random.random()<0.5:
            demand.append((random.uniform(0.25*r,0.5*r),random.uniform(0.05*r,0.01*r)))
        else:
            demand.append((random.uniform(0.05*r,0.01*r),random.uniform(0.25*r,0.5*r)))

    return edges,duration,demand,position



class MyEnv(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        self.M = 10                                                                                 #动作列表长度，不包括-1
        self.t_unit = 10                                                                            #时间单位，用于生成DAG节点的任务需求时间，80%的概率平均分布于t-3t 20的概率平均分布在10t-15t
        self.cpu_res_unit = 100                                                                          #CPU资源单位，用于生成DAG节点的CPU占用需求，50%的概率的任务为CPU资源需求密集型，随机占用0.25r-0.5r 50%的概率随机占用0.05r-0.01r
        self.memory_res_unit = 100                                                                       #Memory资源单位，用于生成DAG节点的memoory占用需求，50%的概率的任务为memory资源需求密集型，随机占用0.25r-0.5r 50%的概率随机占用0.05r-0.01r
        #以下是各状态的上限定义，好像gym中都需要
        self.max_time = np.array([[999999]],dtype=np.float32)                                       #当前执行时间/当前执行时间上限：没有规定上限，理论上是inf
        self.backlot_max_time = np.array([[999999]],dtype=np.float32)                               #backlot中任务所需要总时间/backlot中任务所需要总时间上限：没有规定上限，理论上是inf     //超过动作列表长度的任务的信息被算进backlot，backlot没有个数上限
        self.max_cpu_res=  np.array([[self.cpu_res_unit]],dtype=np.float32)                              #计算资源中的CPU总容量
        self.max_memory_res=  np.array([[self.memory_res_unit]],dtype=np.float32)                        #计算资源中的Memory总容量
        self.t_duration = self.t_unit * np.ones((1,self.M),dtype=np.float32)                        #动作列表中每一个任务的执行时间/动作列表中每一个任务的执行时间上限
        self.cpu_res_limt = self.cpu_res_unit* 0.5 * np.ones((1,self.M),dtype=np.float32)                #动作列表中每一个任务CPU资源需求/动作列表中每一个任务CPU资源需求上限
        self.backlot_cpu_res_limt = np.array([[100 * self.cpu_res_unit * 0.5]],dtype=np.float32)         #backlot中任务所需要总CPU资源
        self.memory_res_limt = self.memory_res_unit* 0.5 * np.ones((1,self.M),dtype=np.float32)          #动作列表中每一个任务memory资源需求/动作列表中每一个任务memory资源需求上限
        self.backlot_memory_res_limt = np.array([[100 * self.memory_res_unit * 0.5]],dtype=np.float32)   #backlot中任务所需要总memory资源
        self.max_b_level = np.array([[100]],dtype=np.float32)                                       #b-level上限
        self.max_children = np.array([[100]],dtype=np.float32)                                      #max_children上限
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
       
        self.seed()
        self.viewer = None
        self.state = None


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
        if (task_cpu_demand != -1)&(task_memory_demand != -1):
            return True if (task_cpu_demand <= self.cpu_res) & (task_memory_demand <= self.memory_res) else False
        else: 
            return False

    def check_episode_finish(self):
        '''
        判断当前幕是否已经执行完成
        :para None
        :return: True 完成一幕了 False 还未完成一幕
        '''
        return True if len(self.done_job) == len(self.duration)  else False


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        '''
        状态转移过程
        :para action {-1,0,1,2,3,4,5,6,7,8,9}
        :return: 下一个状态，回报，是否完成，调试信息
                 False ，要求重新采样动作
        '''
        if action >= 0:                                                                                             #如果选择schedule task
            if self.res_is_available(action):                                                                       #先判断是否为有效动作，如果是：
                self.tasks.append(self.ready_list[action])#self.ready_list[action]表示的是任务ID                                                          #将当前任务ID添加到挂起的任务列表中
                self.tasks_remaing_time[self.ready_list[action]] = self.duration[self.ready_list[action]-1]         #将当前任务的剩余时间添加进字典
                self.cpu_res -= self.cpu_demand[action]                                                             #当前CPU资源容量-这个任务需求的CPU资源
                self.memory_res -= self.memory_demand[action]                                                       #当前Memory资源容量-这个任务需求的memory资源
                self.time = self.time                                                                               #时间步未动-->>其他状态都没变
                self.state = [self.time, self.cpu_res, self.memory_res] + self.wait_duration + self.cpu_demand + \
                    self.memory_demand + [self.b_level, self.children_num, self.backlot_time, self.backlot_cpu_res, self.backlot_memory_res]                                          
                reward = 0                                                                                          #时间步没动，收益为0
                done = self.check_episode_finish()                                                                  #判断这一幕是否完成
                return np.array(self.state, dtype=np.float32), reward, done, self.tasks_remaing_time
            else:                                                                                                   #如果不是，则返回False，要求重新采样动作
                return False
        else:              
            if self.tasks:                                                                                              #如果选择execute task,看当前计算资源上是否有挂起的任务
                self.tasks_remaing_time = sorted(self.tasks_remaing_time.items(),key = lambda x:x[1])                   #将字典的value从小到大排序，排序当前挂起任务的执行占用时间
                self.time += self.tasks_remaing_time[0][1]#self.tasks_remaing_time是任务ID和任务占用时间键值对              #环境时钟增加当前挂起任务中耗时最短的任务的时间
                self.done_job.append(self.tasks_remaing_time[0][0])                                                     #将执行完成的任务加入done_job列表
                self.cpu_res += self.cpu_demand[self.tasks_remaing_time[0][0]-1]#任务ID在列表中索引要-1                    #释放CPU资源
                self.memory_res += self.memory_demand[self.tasks_remaing_time[0][0]-1]#任务ID在列表中索引要-1              #释放Memory资源
                reward = -self.tasks_remaing_time[0][1]                                                                 #在删除前先计算reward
                self.tasks.remove(self.tasks_remaing_time[0][0])                                                        #从计算资源挂起任务中删除此项任务                                                                
                del self.tasks_remaing_time[0]                                                                          #删除该任务ID及其占用时间
                self.ready_list = self.update_ready_list(self.ready_list,self.done_job,self.edges[:])                   #更新ready任务列表
                #刷新以下数据
                self.wait_duration = [-1] * self.M                                                                      #随机生成的DAG时间占用信息
                self.cpu_demand = [-1] * self.M                                                                         #随机生成的DAG资源占用信息
                self.memory_demand = [-1] * self.M  
                self.backlot_time = 0                                                                                   #backlot的总时间占用信息
                self.backlot_cpu_res = 0                                                                                #backlot的总CPU占用信息 
                self.backlot_memory_res = 0                                                                             #backlot的总memory占用信息 
                for i in range(len(self.ready_list)):                                   
                    if i < self.M: 
                        self.wait_duration[i] = self.duration[self.ready_list[i]-1]                                       
                        self.cpu_demand[i] = self.demand[self.ready_list[i]-1][0]
                        self.memory_demand[i] = self.demand[self.ready_list[i]-1][1]
                    else:
                        self.backlot_time += self.duration[self.ready_list[i]-1]
                        self.backlot_cpu_res += self.demand[self.ready_list[i]-1][0]
                        self.backlot_memory_res += self.demand[self.ready_list[i]-1][1]
                self.b_level = self.find_b_level(self.edges[:],self.done_job)                                           #更新b_level    
                self.children_num = self.find_children_num(self.ready_list,self.edges[:])                               #更新children_num
                self.state = [self.time, self.cpu_res, self.memory_res] + self.wait_duration + self.cpu_demand + \
                        self.memory_demand + [self.b_level, self.children_num, self.backlot_time, self.backlot_cpu_res, self.backlot_memory_res]                                                                 #reward等于负的时间前进步
                done = self.check_episode_finish()                                                                      #判断是否完成
                return np.array(self.state, dtype=np.float32), reward, done, self.tasks_remaing_time
            else:                                                                                                       #如果没有，则返回false重新采样动作
                return False


    def reset(self):
        '''
        初始化状态
        :para None
        :return: 初始状态，每一次reset就是重新一幕
        '''
        ###初始化一些状态 
        self.edges,self.duration,self.demand,self.position = workflows_generator('default')
        print("任务占用时间Ti:",self.duration)
        print("任务资源占用(res_cpu,res_memory):",self.demand)
        self.time = 0
        self.cpu_res = 100
        self.memory_res = 100
        self.done_job = []
        self.ready_list =[]
        self.wait_duration = [-1] * self.M                      #随机生成的DAG时间占用信息
        self.cpu_demand = [-1] * self.M                         #随机生成的DAG资源占用信息
        self.memory_demand = [-1] * self.M  
        self.backlot_time = 0                                   #backlot的总时间占用信息
        self.backlot_cpu_res = 0                                #backlot的总CPU占用信息 
        self.backlot_memory_res = 0                             #backlot的总memory占用信息 

        self.ready_list = self.update_ready_list(self.ready_list,self.done_job,self.edges[:])
        for i in range(len(self.ready_list)):
            if i < self.M: 
                self.wait_duration[i] = self.duration[self.ready_list[i]-1]
                self.cpu_demand[i] = self.demand[self.ready_list[i]-1][0]
                self.memory_demand[i] = self.demand[self.ready_list[i]-1][1]
            else:
                self.backlot_time += self.duration[self.ready_list[i]-1]
                self.backlot_cpu_res += self.demand[self.ready_list[i]-1][0]
                self.backlot_memory_res += self.demand[self.ready_list[i]-1][1]


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
