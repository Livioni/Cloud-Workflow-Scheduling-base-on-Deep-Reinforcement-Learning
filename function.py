from DAGs_Generator import workflows_generator

# edges,duration,demand = workflows_generator('default')

edges = [(1, 5), (1, 4), (2, 5), (3, 4), (4, 9), (4, 6), (5, 10), ('Start', 1), ('Start', 2), ('Start', 3), ('Start', 7), ('Start', 8), (6, 'Exit'), (7, 'Exit'), (8, 'Exit'), (9, 'Exit'), (10, 'Exit')]

#寻找前继节点
def search_for_predecessor(node,edges):
    map = {}
    for i in range(len(edges)):
        if edges[i][0] in map.keys():
            map[edges[i][0]].append(edges[i][1])
        else:
            map[edges[i][0]] = [edges[i][1]]
    pred = map[node]
    return pred

#寻找后续节点
def search_for_successors(node,edges):
    map = {}
    for i in range(len(edges)):
        if edges[i][1] in map.keys():
            map[edges[i][1]].append(edges[i][0])
        else:
            map[edges[i][1]] = [edges[i][0]]
    succ = map[node]
    return succ


