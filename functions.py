from DAGs_Generator import workflows_generator

# edges,duration,demand = workflows_generator('default')

edges = [(1, 5), (1, 4), (2, 5), (3, 4), (4, 9), (4, 6), (5, 10), ('Start', 1), ('Start', 2), ('Start', 3), ('Start', 7), ('Start', 8), (6, 'Exit'), (7, 'Exit'), (8, 'Exit'), (9, 'Exit'), (10, 'Exit')]
ready_list = []
done_job = []
#寻找前继节点
def search_for_predecessor(node,edges):
    map = {}
    if node == 'Start': return print("error, 'Start' node do not have predecessor!")
    for i in range(len(edges)):
        if edges[i][1] in map.keys():
            map[edges[i][1]].append(edges[i][0])
        else:
            map[edges[i][1]] = [edges[i][0]]
    succ = map[node]
    return succ

#寻找后续节点
def search_for_successors(node,edges):
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

print(search_for_successors('Exit',edges))