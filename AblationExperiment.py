import gym, torch, copy, os, xlwt, random
import torch.nn as nn
from datetime import datetime
import numpy as np

env = gym.make("clusterEnv-v0").unwrapped
state_dim, action_dim = env.return_dim_info()

####### initialize environment hyperparameters ######
max_ep_len = 1000  # max timesteps in one episode
auto_save = 1
total_test_episodes = 100 * auto_save  # total num of testing episodes


def initial_excel():
    global worksheet, workbook
    # xlwt 库将数据导入Excel并设置默认字符编码为ascii
    workbook = xlwt.Workbook(encoding='ascii')
    # 添加一个表 参数为表名
    worksheet = workbook.add_sheet('makespan')
    # 生成单元格样式的方法
    # 设置列宽, 3为列的数目, 12为列的宽度, 256为固定值
    for i in range(3):
        worksheet.col(i).width = 256 * 12
    # 设置单元格行高, 25为行高, 20为固定值
    worksheet.row(1).height_mismatch = True
    worksheet.row(1).height = 20 * 25
    # 保存excel文件
    workbook.save('data/makespan_MCTSAE.xls')


def read_current_state():
    '''
    读取当前env的状态
    :return: 当前env的状态
    '''
    state = copy.deepcopy(env.state)
    ready_list = copy.deepcopy(env.ready_list)
    done_job = copy.deepcopy(env.done_job)
    tasks = copy.deepcopy(env.tasks)
    wait_duration = copy.deepcopy(env.wait_duration)
    cpu_demand = copy.deepcopy(env.cpu_demand)
    memory_demand = copy.deepcopy(env.memory_demand)
    tasks_remaing_time = copy.deepcopy(env.tasks_remaing_time)
    time = env.time
    cpu_res = env.cpu_res
    memory_res = env.memory_res
    return state, ready_list, done_job, tasks, wait_duration, cpu_demand, memory_demand, tasks_remaing_time, cpu_res, memory_res, time


def load_current_state(state, ready_list, done_job, tasks, wait_duration, cpu_demand, memory_demand, tasks_remaing_time,
                       cpu_res, memory_res, time):
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
    env.set_time(time)
    return


class TreeNode(object):
    def __init__(self, parent, state, ready_list, done_job, tasks, wait_duration, cpu_demand, memory_demand,
                 tasks_remaing_time, cpu_res, memory_res, time):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._makespan = 0
        self._total_makespan = 0
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
        self._time = time
        self._c = 40
        self._value = 0
        if self._parent != None:
            self.get_value()

    def expand(self):
        '''
        扩展树
        '''
        load_current_state(self._state, self._ready_list, self._done_job, self._tasks, self._wait_duration,
                           self._cpu_demand, self._memory_demand, self._tasks_remaing_time, self._cpu_res,
                           self._memory_res, self._time)
        available_action = env.return_action_list()
        if available_action:
            for action in available_action:
                load_current_state(self._state, self._ready_list, self._done_job, self._tasks, self._wait_duration,
                                   self._cpu_demand, self._memory_demand, self._tasks_remaing_time, self._cpu_res,
                                   self._memory_res, self._time)
                if action not in self._children:
                    env.step(action)
                    state, ready_list, done_job, tasks, wait_duration, cpu_demand, memory_demand, tasks_remaing_time, cpu_res, memory_res, time = read_current_state()
                    self._children[action] = TreeNode(self, state, ready_list, done_job, tasks, wait_duration,
                                                      cpu_demand, memory_demand, tasks_remaing_time, cpu_res,
                                                      memory_res, time)
        else:
            print("done")

    def get_average_makespan(self):
        return self._makespan

    def get_value(self):
        self._value = self._makespan + self._c * np.sqrt(np.log(self._parent._n_visits + 1) / (self._n_visits + 1))
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
        if self._parent != None:
            self._value = self.get_value()

    def update_recursive(self, leaf_value):
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    def __init__(self, state, ready_list, done_job, tasks, wait_duration, cpu_demand, memory_demand, tasks_remaing_time,
                 cpu_res, memory_res, time, depth):
        self._root = TreeNode(None, state, ready_list, done_job, tasks, wait_duration, cpu_demand, memory_demand,
                              tasks_remaing_time, cpu_res, memory_res, time)
        self._root.expand()  # 初始化扩展
        self._initial_buget = 100
        self._min_buget = 10
        self._depth = depth

    def playout(self):
        buget = max(self._initial_buget / self._depth, self._min_buget)
        for j in range(int(buget)):
            node = self._root
            while True:
                if node.is_leaf():
                    if node._n_visits == 0:
                        cur_state, cur_ready_list, cur_done_job, cur_tasks, cur_wait_duration, cur_cpu_demand, cur_memory_demand, cur_tasks_remaing_time, cur_cpu_res, cur_memory_res, cur_time = node._state, node._ready_list, node._done_job, node._tasks, node._wait_duration, node._cpu_demand, node._memory_demand, node._tasks_remaing_time, node._cpu_res, node._memory_res, node._time
                        makespan = self._roll_out(cur_state, cur_ready_list, cur_done_job, cur_tasks, cur_wait_duration,
                                                  cur_cpu_demand, cur_memory_demand, cur_tasks_remaing_time,
                                                  cur_cpu_res, cur_memory_res, cur_time)
                        node.update_recursive(makespan)
                        break
                    else:
                        node.expand()
                        node = node.select()
                else:
                    node = node.select()
        node = self._root
        return max(node._children.items(), key=lambda act_node: act_node[1].get_average_makespan())[0]

    def _roll_out(self, cur_state, cur_ready_list, cur_done_job, cur_tasks, cur_wait_duration, cur_cpu_demand,
                  cur_memory_demand, cur_tasks_remaing_time, cur_cpu_res, cur_memory_res, cur_time):
        load_current_state(cur_state, cur_ready_list, cur_done_job, cur_tasks, cur_wait_duration, cur_cpu_demand,
                           cur_memory_demand, cur_tasks_remaing_time, cur_cpu_res, cur_memory_res, cur_time)
        state = cur_state
        max_ep_len = 1000  # max timesteps in one episode
        for t in range(1, max_ep_len + 1):
            action = random.choice(range(action_dim)) - 1
            state, reward, done, info = env.step(action)
            while (info[0] == False):
                action = random.choice(range(action_dim)) - 1
                state, reward, done, info = env.step(action)  # 输入step的都是
            next_state, reward, done, _ = state, reward, done, info
            # break; if the episode is over
            state = next_state
            if done:
                makespan = state[0]
                break
        return makespan


if __name__ == '__main__':
    initial_excel()
    makespans = []
    line = 0
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("============================================================================================")
    for ep in range(1, total_test_episodes + 1):
        initial_state = env.reset()
        state, ready_list, done_job, tasks, wait_duration, cpu_demand, memory_demand, tasks_remaing_time, cpu_res, memory_res, time = read_current_state()
        for depth in range(1, max_ep_len + 1):
            tree = MCTS(state, ready_list, done_job, tasks, wait_duration, cpu_demand, memory_demand,
                        tasks_remaing_time, cpu_res, memory_res, time, depth=depth)
            best_action = tree.playout()
            load_current_state(tree._root._state, tree._root._ready_list, tree._root._done_job, tree._root._tasks,
                               tree._root._wait_duration, tree._root._cpu_demand, tree._root._memory_demand,
                               tree._root._tasks_remaing_time, tree._root._cpu_res, tree._root._memory_res,
                               tree._root._time)
            observation, reward, done, info = env.step(best_action)
            state, ready_list, done_job, tasks, wait_duration, cpu_demand, memory_demand, tasks_remaing_time, cpu_res, memory_res, time = read_current_state()
            del tree
            if done:
                makespan = observation[0]
                makespans.append(makespan)
                print("Episode:", ep, "Makespan:", makespan)
                if ep % auto_save == 0:
                    average_makespan = np.mean(makespans)
                    worksheet.write(line, 1, float(average_makespan))
                    workbook.save('data/makespan_MCTSAE.xls')
                    print('MCTS : Episode: {},  Makespan: {:.3f}s'.format((line + 1) * auto_save, average_makespan))
                    line += 1
                    makespans = []
                    end_time = datetime.now().replace(microsecond=0)
                    print("Finished testing at (GMT) : ", end_time)
                    print("Total testing time  : ", end_time - start_time)
                    start_time = end_time
                break
            workbook.save('data/makespan_MCTSAE.xls')
    env.close()
