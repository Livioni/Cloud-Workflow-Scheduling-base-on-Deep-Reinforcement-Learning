import argparse
import math
import random
from scipy import sparse
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from numpy.random.mtrand import sample

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='default', type=str)  # parameters setting
parser.add_argument('--n', default=10, type=int)  # number of DAG  nodes
parser.add_argument('--max_out', default=2, type=float)  # max out_degree of one node
parser.add_argument('--alpha', default=1, type=float)  # shape
parser.add_argument('--beta', default=1.0, type=float)  # regularity
args = parser.parse_args()

set_dag_size = [20, 30, 40, 50, 60, 70, 80, 90]  # random number of DAG  nodes
set_max_out = [1, 2, 3, 4, 5]  # max out_degree of one node
set_alpha = [0.5, 1.0, 1.5]  # DAG shape
set_beta = [0.0, 0.5, 1.0, 2.0]  # DAG regularity


def DAGs_generate(mode='default', n=10, max_out=2, alpha=1, beta=1.0):
    ##############################################initialize############################################
    if mode != 'default':
        args.n = random.sample(set_dag_size, 1)[0]
        args.max_out = random.sample(set_max_out, 1)[0]
        args.alpha = random.sample(set_alpha, 1)[0]
        args.beta = random.sample(set_beta, 1)[0]
        args.prob = 0.9
    else:
        args.n = n
        args.max_out = random.sample(set_max_out, 1)[0]
        args.alpha = random.sample(set_alpha, 1)[0]
        args.beta = random.sample(set_beta, 1)[0]
        args.prob = 1

    length = math.floor(math.sqrt(args.n) / args.alpha)  # 根据公式计算出来的DAG深度
    mean_value = args.n / length  # 计算平均长度
    random_num = np.random.normal(loc=mean_value, scale=args.beta, size=(length, 1))  # 计算每一层的数量，满足正态分布
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

    if generate_num != args.n:  # 不等的话做微调
        if generate_num < args.n:
            for i in range(args.n - generate_num):
                index = random.randrange(0, length, 1)
                dag_list[index].append(len(dag_list[index]))
        if generate_num > args.n:
            i = 0
            while i < generate_num - args.n:
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
    into_degree = [0] * args.n
    out_degree = [0] * args.n
    edges = []
    pred = 0

    for i in range(length - 1):
        sample_list = list(range(len(dag_list_update[i + 1])))
        for j in range(len(dag_list_update[i])):
            od = random.randrange(1, args.max_out + 1, 1)
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
        if random.random() < args.prob:
            # duration.append(random.uniform(t,3*t))
            duration.append(random.sample(range(0, 3 * t), 1)[0])
        else:
            # duration.append(random.uniform(5*t,10*t))
            duration.append(random.sample(range(5 * t, 10 * t), 1)[0])
    # 初始化资源需求
    # for i in range(len(in_degree)):
    #     if random.random() < 0.5:
    #         demand.append((random.uniform(0.25 * r, 0.5 * r), random.uniform(0.05 * r, 0.01 * r)))
    #     else:
    #         demand.append((random.uniform(0.05 * r, 0.01 * r), random.uniform(0.25 * r, 0.5 * r)))
    for i in range(len(in_degree)):
        demand.append((random.uniform(0.25 * r, 0.5 * r), random.uniform(0.25 * r, 0.5 * r)))

    return edges, duration, demand, position


def generate_train_datasheet(amount, DAGsize):
    edges_lib = []
    duration_lib = []
    demand_lib = []
    # 生成amount个随机的DAG图
    for ele in range(0, amount):
        edges, duration, demand, _ = workflows_generator('default', n=DAGsize)
        edges_lib.append(edges)
        duration_lib.append(duration)
        demand_lib.append(demand)

    edges_lib_np = np.array(edges_lib,dtype=object)
    duration_lib_np = np.array(duration_lib, dtype=np.float32)
    demand_lib_np = np.array(demand_lib, dtype=np.float32)

    np.save('npy/train_datasheet/' + str(args.n) + '/edges' + str(args.n) + '_lib.npy', edges_lib_np)
    np.save('npy/train_datasheet/' + str(args.n) + '/duration' + str(args.n) + '_lib.npy', duration_lib_np)
    np.save('npy/train_datasheet/' + str(args.n) + '/demand' + str(args.n) + '_lib.npy', demand_lib_np)


def generate_test_datasheet(amount, DAGsize):
    edges_lib = []
    duration_lib = []
    demand_lib = []
    # 生成amount个随机的DAG图
    for ele in range(0, amount):
        edges, duration, demand, _ = workflows_generator('default', n=DAGsize)
        edges_lib.append(edges)
        duration_lib.append(duration)
        demand_lib.append(demand)

    edges_lib_np = np.array(edges_lib,dtype=object)
    duration_lib_np = np.array(duration_lib, dtype=np.float32)
    demand_lib_np = np.array(demand_lib, dtype=np.float32)

    np.save('npy/test_datasheet/' + str(args.n) + '/edges' + str(args.n) + '_lib.npy', edges_lib_np)
    np.save('npy/test_datasheet/' + str(args.n) + '/duration' + str(args.n) + '_lib.npy', duration_lib_np)
    np.save('npy/test_datasheet/' + str(args.n) + '/demand' + str(args.n) + '_lib.npy', demand_lib_np)


if __name__ == '__main__':
    generate_train_datasheet(1000, 10)
    generate_test_datasheet(1000,10)
    # edges, duration, demand, _ = workflows_generator('default')