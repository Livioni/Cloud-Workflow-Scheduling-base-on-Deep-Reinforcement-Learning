from networkx.algorithms import dominating
from DAGs_Generator import workflows_generator
import networkx as nx

M = 5

# edges,duration,demand = workflows_generator('default')
edges = [(1, 7), (1, 6), (2, 7), (3, 7), (4, 6), (4, 7), (5, 6), (5, 7), (6, 9), (7, 8), (7, 10), ('Start', 1), ('Start', 2), ('Start', 3), ('Start', 4), ('Start', 5), (8, 'Exit'), (9, 'Exit'), (10, 'Exit')]
duration = [12.093872682578908, 19.220424108680735, 29.805698906159055, 22.722194353509664, 26.101160966227223, 22.935788547458667, 26.351666691590356, 15.039301662808857, 21.44768962005825, 21.55420388678772]
demand = [(30.757662513101025, 3.167983697298989), (3.1492983091761593, 37.41426693020626), (2.363107815153717, 34.19981368748465), (31.23791132822727, 2.1912999829977053), (4.5450154830068, 37.995899321938474), (37.350404493621696, 4.577078218529003), (40.56363387729358, 2.9232817576682493), (2.940579646707287, 48.48645463208979), (47.90208901784828, 3.4026323412020227), (38.853838716347894, 4.3424621230597555)]
ready_list = []
available_list = []*M
done_job = []

def search_for_predecessor(node,edges):
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


def search_for_successors(node,edges):
    '''
    寻找后续节点
    :param node: 需要查找的节点id
    :param edges: DAG边信息(注意最好传列表的值（edges[:]）进去而不是传列表的地址（edges）！！！！)
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

def update_ready_list(ready_list,done_job,edges):
    '''
    根据已完成的任务更新当前可以执行的task列表，满足DAG的依赖关系。并不表明它可以被执行，因为还要受资源使用情况限制
    :param ready_list: 上一时刻可用task列表
    :param done_job: DAG中已完成的任务id列表
    :para edges: DAG边信息(注意最好传列表的值（edges[:]）进去而不是传列表的地址（edges）！！！！)
    :return: 更新完成后的task列表
    '''
    all_succ = []
    preds = []
    if ready_list:
        for i in range(len(ready_list)):
            all_succ.extend(search_for_successors(ready_list[i],edges))
    else:
        ready_list = ['Start']
        return update_ready_list(ready_list,['Start'],edges)
    for i in all_succ:
        preds = search_for_predecessor(i,edges)
        if (set(done_job)>= set(preds)):
            ready_list.append(i)
    for job in done_job : 
        if job in ready_list: ready_list.remove(job)
    return sorted(set(ready_list), key = ready_list.index)

def find_b_level(edges,done_job):
    '''
    计算当前未完成DAG的b-level（longest path) 不包括 Start 和 Exit节点
    :para edges: DAG边信息(注意最好传列表的值（edges[:]）进去而不是传列表的地址（edges）！！！！)
    :param done_job: DAG中已完成的任务id列表 注意done_job是所有任务id的子集
    :return: 更新完成后的task列表
    '''
    for i in range(len(done_job)):
        preds = search_for_predecessor(done_job[i],edges)
        for j in preds: edges.pop(edges.index((j,done_job[i])))
    g1 = nx.DiGraph()
    g1.add_edges_from(edges)
    b_level_road = nx.dag_longest_path(g1)
    return b_level_road[1:-1]

def find_children_num(ready_list,edges):
    '''
    计算ready task的子节点数，作为状态的输入
    :param ready_list: DAG中准备好的任务列表
    :para edges: DAG边信息(注意最好传列表的值（edges[:]）进去而不是传列表的地址（edges）！！！！)
    :return: 子节点的数量，不包括'Exit'
    '''
    children = []
    for job in ready_list:
        children.extend(search_for_successors(job,edges))
    length = 0
    while (length != len(children)):
        length = len(children)
        for job in children:
            if job != 'Exit':
                children.extend(search_for_successors(job,edges))
            children = sorted(set(children), key = children.index)
    return len(set(children))-1     

ready_list = [1,3,4,5]
done_job = [1]
ready_list = update_ready_list(ready_list,done_job,edges[:])
print(ready_list)