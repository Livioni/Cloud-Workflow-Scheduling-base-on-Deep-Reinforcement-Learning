import gym, random, copy, xlwt
from gym import spaces
from gym.utils import seeding
import numpy as np
import networkx as nx
from . import utils


class clusterEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        self.M = 30  # 动作列表长度，不包括-1
        self.t_unit = 10  # 时间单位，用于生成DAG节点的任务需求时间，80%的概率平均分布于t-3t 20的概率平均分布在10t-15t
        self.cpu_res_unit = 100  # CPU资源单位，用于生成DAG节点的CPU占用需求，50%的概率的任务为CPU资源需求密集型，随机占用0.25r-0.5r 50%的概率随机占用0.05r-0.01r
        self.memory_res_unit = 100  # Memory资源单位，用于生成DAG节点的memoory占用需求，50%的概率的任务为memory资源需求密集型，随机占用0.25r-0.5r 50%的概率随机占用0.05r-0.01r
        self.time, self.cpu_res, self.memory_res = 0, 100, 100
        ##状态信息
        self.time = 0  # 整体环境时钟，DAG执行已花费时间
        self.cpu_res = 100  # 当前计算资源的CPU容量
        self.memory_res = 100  # 当前计算资源的memory容量
        self.b_level = None  # b-level值
        self.children_num = None  # 子节点数
        self.edges = []  # 随机生成DAG的信息
        self.available_list = [-1] * self.M  # 调度输入个数 尽量大于readylist 取self.M，不足补-1
        self.duration = []  # 随机生成DAG的时间占用信息
        self.demand = []  # backlot的总资源占用信息
        self.wait_duration = [-1] * self.M  # 随机生成的DAG时间占用信息
        self.cpu_demand = [-1] * self.M  # 随机生成的DAG cpu 资源占用信息
        self.memory_demand = [-1] * self.M  # 随机生成的DAG memory 资源占用信息
        self.ready_list = []  # 满足依赖关系的DAG,可能被执行，但不满足
        self.done_job = []  # 已完成的任务ID
        self.position = []  # 当前DAG画图坐标
        self.backlot_time = 0  # backlot的总时间占用信息
        self.backlot_cpu_res = 0  # backlot的总CPU占用信息
        self.backlot_memory_res = 0  # backlot的总memory占用信息
        ##状态转移信息和中间变量
        self.tasks = []  # 计算资源上挂起的任务
        self.tasks_remaing_time = {}  # 计算资源上挂起的任务剩余执行时间

        self.seed1 = 0
        self.viewer = None
        self.state = None
        self.edges_lib = []
        self.duration_lib = []
        self.demand_lib = []

        self.DAGsize = 30
        self.load_train_dataset(self.DAGsize)
        # self.load_test_dataset(self.DAGsize)

    def set_state(self, state):
        self.state = copy.deepcopy(state)

    def set_ready_list(self, ready_list):
        self.ready_list = copy.deepcopy(ready_list)

    def set_done_job(self, done_job):
        self.done_job = copy.deepcopy(done_job)

    def set_tasks(self, tasks):
        self.tasks = copy.deepcopy(tasks)

    def set_wait_duration(self, wait_duration):
        self.wait_duration = copy.deepcopy(wait_duration)

    def set_cpu_demand(self, cpu_demand):
        self.cpu_demand = copy.deepcopy(cpu_demand)

    def set_memory_demand(self, memory_demand):
        self.memory_demand = copy.deepcopy(memory_demand)

    def set_tasks_remaing_time(self, tasks_remaing_time):
        self.tasks_remaing_time = copy.deepcopy(tasks_remaing_time)

    def set_cpu_res(self, cpu_res):
        self.cpu_res = cpu_res

    def set_memory_res(self, memory_res):
        self.memory_res = memory_res

    def set_time(self, time):
        self.time = time

    def load_train_dataset(self, DAGsize):
        ##########################################training################################
        print('train datasheet lib.')
        edges_lib_path = '/Users/livion/Documents/GitHub/Cloud-Workflow-Scheduling-base-on-Deep-Reinforcement-Learning/npy/train_datasheet/' + str(
            DAGsize) + '/edges' + str(DAGsize) + '_lib.npy'
        duration_lib_path = '/Users/livion/Documents/GitHub/Cloud-Workflow-Scheduling-base-on-Deep-Reinforcement-Learning/npy/train_datasheet/' + str(
            DAGsize) + '/duration' + str(DAGsize) + '_lib.npy'
        demand_lib_path = '/Users/livion/Documents/GitHub/Cloud-Workflow-Scheduling-base-on-Deep-Reinforcement-Learning/npy/train_datasheet/' + str(
            DAGsize) + '/demand' + str(DAGsize) + '_lib.npy'
        # edges_lib_path = '/Users/livion/Documents/GitHub/Cloud-Workflow-Scheduling-base-on-Deep-Reinforcement-Learning/npy/train_datasheet/'+'test'+'/edges' +'test' +'_lib.npy'
        # duration_lib_path = '/Users/livion/Documents/GitHub/Cloud-Workflow-Scheduling-base-on-Deep-Reinforcement-Learning/npy/train_datasheet/'+'test'+'/duration' + 'test' +'_lib.npy'
        # demand_lib_path = '/Users/livion/Documents/GitHub/Cloud-Workflow-Scheduling-base-on-Deep-Reinforcement-Learning/npy/train_datasheet/'+'test'+'/demand'+'test'+'_lib.npy'
        self.edges_lib = np.load(edges_lib_path, allow_pickle=True).tolist()
        self.duration_lib = np.load(duration_lib_path, allow_pickle=True).tolist()
        self.demand_lib = np.load(demand_lib_path, allow_pickle=True).tolist()
        print('load completed.')
        return

    def load_test_dataset(self, DAGsize):
        #########################################testing################################
        print('test datasheet loaded.')
        edges_lib_path = '/Users/livion/Documents/GitHub/Cloud-Workflow-Scheduling-base-on-Deep-Reinforcement-Learning/npy/test_datasheet/' + str(
            DAGsize) + '/edges' + str(DAGsize) + '_lib.npy'
        duration_lib_path = '/Users/livion/Documents/GitHub/Cloud-Workflow-Scheduling-base-on-Deep-Reinforcement-Learning/npy/test_datasheet/' + str(
            DAGsize) + '/duration' + str(DAGsize) + '_lib.npy'
        demand_lib_path = '/Users/livion/Documents/GitHub/Cloud-Workflow-Scheduling-base-on-Deep-Reinforcement-Learning/npy/test_datasheet/' + str(
            DAGsize) + '/demand' + str(DAGsize) + '_lib.npy'
        self.edges_lib = np.load(edges_lib_path, allow_pickle=True).tolist()
        self.duration_lib = np.load(duration_lib_path, allow_pickle=True).tolist()
        self.demand_lib = np.load(demand_lib_path, allow_pickle=True).tolist()
        print('load completed.')
        return

    def return_dim_info(self):
        return (3 + self.M * 3 + 5), self.M + 1

    def _search_for_predecessor(self, node, edges):
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

    def _search_for_successors(self, node, edges):
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

    def _update_ready_list(self, ready_list, done_job, edges):
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
                all_succ.extend(self._search_for_successors(ready_list[i], edges))
        else:
            ready_list = ['Start']
            return self._update_ready_list(ready_list, ['Start'], edges)
        for i in all_succ:
            preds = self._search_for_predecessor(i, edges)
            if (set(done_job) >= set(preds)):
                ready_list.append(i)
        for job in done_job:
            if job in ready_list: ready_list.remove(job)
        return sorted(set(ready_list), key=ready_list.index)

    def _find_b_level(self, edges, done_job):
        '''
        计算当前未完成DAG的b-level（lenth of the longest path) 不包括 Start 和 Exit节点
        :para edges: DAG边信息(注意最好传列表的值（edges[:]）进去而不是传列表的地址（edges）!!!)
        :param done_job: DAG中已完成的任务id列表 注意done_job是所有任务id的子集
        :return: b_level
        '''
        for i in range(len(done_job)):
            preds = self._search_for_predecessor(done_job[i], edges)
            for j in preds: edges.pop(edges.index((j, done_job[i])))
        g1 = nx.DiGraph()
        g1.add_edges_from(edges)
        b_level_road = nx.dag_longest_path(g1)
        return len(b_level_road[1:-1])

    def _find_children_num(self, ready_list, edges):
        '''
        计算ready task的子节点数，作为状态的输入
        :param ready_list: DAG中准备好的任务列表
        :para edges: DAG边信息(注意最好传列表的值（edges[:]）进去而不是传列表的地址（edges）!!!)
        :return: 子节点的数量，不包括'Exit'
        '''
        children = []
        for job in ready_list:
            children.extend(self._search_for_successors(job, edges))
        length = 0
        while (length != len(children)):
            length = len(children)
            for job in children:
                if job != 'Exit':
                    children.extend(self._search_for_successors(job, edges))
                children = sorted(set(children), key=children.index)
        return len(set(children)) - 1

    def res_is_available(self, action):
        '''
        判断当前选择的动作是否能在计算资源上执行
        :para action {-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29}
        :return: True 可以被执行 False 不能被执行
        '''
        task_cpu_demand = self.cpu_demand[action]
        task_memory_demand = self.memory_demand[action]
        if self.wait_duration[action] != -1.0:
            return True if (task_cpu_demand <= self.cpu_res) & (
                    task_memory_demand <= self.memory_res) else False  # 判断资源是否满足要求
        else:
            return False

    def check_action(self, action):
        '''
        判断当前选择的动作是否有效
        :para action {-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29}
        :return: True 有效 False 无效
        '''
        if (action < len(self.ready_list)):
            if (self.ready_list[action] not in self.done_job):
                if (self.ready_list[action] not in self.tasks):
                    return self.res_is_available(action)
                else:
                    return False
            else:
                return False
        else:
            return False

    def _check_episode_finish(self):
        '''
        判断当前幕是否已经执行完成
        :para None
        :return: True 完成一幕了 False 还未完成一幕
        '''
        if len(self.done_job) >= len(self.duration):
            return True
        else:
            False

    def _pend_task(self, action):
        job_id = self.ready_list[action]
        self.tasks.append(job_id)  # self.ready_list[action]表示的是任务ID
        self.tasks_remaing_time[job_id] = self.wait_duration_dic[job_id]
        self.cpu_res -= self.cpu_demand_dic[job_id]  # 当前CPU资源容量-这个任务需求的CPU资源
        self.memory_res -= self.memory_demand_dic[job_id]
        self.wait_duration[action] = -1.0  # waiting列表中挂起的任务信息变为-1
        self.cpu_demand[action] = -1.0  # waiting列表中挂起的任务信息变为-1
        self.memory_demand[action] = -1.0

    def return_res_usage(self):
        time = self.state[0]
        cpu_usage = self.state[1]
        memory_usage = self.state[2]
        return time, round(cpu_usage, 2), round(memory_usage, 2)

    def return_action_list(self):
        available_action = []
        if self.tasks:
            available_action.append(-1)
        for index in range(len(self.ready_list)):
            if self.check_action(index):
                available_action.append(index)
        return available_action

    def step(self, action):
        '''
        状态转移过程
        :para action {-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29}
        :return: 下一个状态，回报，是否完成，调试信息
                 False ，要求重新采样动作ßå
        '''
        '''
        self.ready_list是当前可执行的任务，包括了在机器上挂起的
        '''
        if (action >= 0 & action <= self.M - 1):
            if (self.check_action(action)):
                self._pend_task(action)
                self.state = [self.time, self.cpu_res, self.memory_res] + self.wait_duration + self.cpu_demand + \
                             self.memory_demand + [self.b_level, self.children_num, self.backlot_time,
                                                   self.backlot_cpu_res, self.backlot_memory_res]
                reward = 0.0  # 时间步没动，收益为0

                done = self._check_episode_finish()
                return np.array(self.state, dtype=np.float32), reward, done, [True, self.tasks]
            else:
                return np.array(self.state, dtype=np.float32), 0, 0, [False, self.tasks]
        else:
            if self.tasks:
                self.tasks_remaing_time_list = sorted(self.tasks_remaing_time.items(),
                                                      key=lambda x: x[1])  # 排序当前挂起任务的执行时间
                job_id = self.tasks_remaing_time_list[0][0]
                time_shift = self.tasks_remaing_time_list[0][1]  # 记录最小执行时间的长度
                self.time += time_shift  # 改变时间戳
                self.done_job.append(job_id)  # 更新已完成的任务
                done = self._check_episode_finish()
                if done:
                    # self.state = [self.time, self.cpu_res, self.memory_res] + self.wait_duration + self.cpu_demand + \
                    #     self.memory_demand + [self.b_level, self.children_num, self.backlot_time, self.backlot_cpu_res, self.backlot_memory_res]
                    return np.array(self.state, dtype=np.float32), 0, done, [True, self.tasks]
                self.cpu_res += self.cpu_demand_dic[job_id]  # 释放CPU资源
                self.memory_res += self.memory_demand_dic[job_id]  # 释放Memory资源
                self.tasks.remove(job_id)  # 删除pending的task
                del self.tasks_remaing_time[job_id]  # 删除任务剩余时间信息
                for i in self.tasks_remaing_time.keys():
                    self.tasks_remaing_time[i] -= time_shift
                reward = -time_shift / self.t_unit
                self.ready_list = self._update_ready_list(self.ready_list, self.done_job, self.edges[:])  # 更新ready_list
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
                self.b_level = self._find_b_level(self.edges[:], self.done_job)
                self.children_num = self._find_children_num(self.ready_list, self.edges[:])
                self.state = [self.time, self.cpu_res, self.memory_res] + self.wait_duration + self.cpu_demand + \
                             self.memory_demand + [self.b_level, self.children_num, self.backlot_time,
                                                   self.backlot_cpu_res, self.backlot_memory_res]
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
        self.seed1 = random.randint(0, len(self.duration_lib) - 1)
        self.edges, self.duration, self.demand = self.edges_lib[self.seed1], self.duration_lib[self.seed1], \
                                                 self.demand_lib[self.seed1]
        # self.seed1 += 1
        # if self.seed1 == 1000:
        #     self.seed1 = 0
        # self.edges,self.duration,self.demand,self.position = utils.workflows_generator('default')
        # print("DAG结构Edges：",self.edges)
        # print("任务占用时间Ti:",self.duration)                    #生成的原始数据
        # print("任务资源占用(res_cpu,res_memory):",self.demand)    #生成的原始数据
        ###初始化一些状态
        self.time, self.cpu_res, self.memory_res = 0, 100, 100
        self.done_job = []
        self.ready_list = []
        self.wait_duration_dic = {}  # 随机生成的DAG时间占用信息
        self.cpu_demand_dic = {}  # 随机生成的DAG CPU资源占用信息
        self.memory_demand_dic = {}  # 随机生成的DAG Memo资源占用信息
        self.backlot_time = 0  # backlot的总时间占用信息
        self.backlot_cpu_res = 0  # backlot的总CPU占用信息
        self.backlot_memory_res = 0  # backlot的总memory占用信息
        self.tasks_remaing_time = {}
        self.tasks = []
        self.ready_list = self._update_ready_list(self.ready_list, self.done_job, self.edges[:])
        for i in range(len(self.duration)):
            self.wait_duration_dic[i + 1] = self.duration[i]
            self.cpu_demand_dic[i + 1] = self.demand[i][0]
            self.memory_demand_dic[i + 1] = self.demand[i][1]
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

        self.b_level = self._find_b_level(self.edges[:], self.done_job)
        self.children_num = self._find_children_num(self.ready_list, self.edges[:])

        self.state = [self.time, self.cpu_res, self.memory_res] + self.wait_duration + self.cpu_demand + \
                     self.memory_demand + [self.b_level, self.children_num, self.backlot_time, self.backlot_cpu_res,
                                           self.backlot_memory_res]

        return np.array(self.state, dtype=np.float32)

    def render(self, mode="human"):
        pass
        return

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
