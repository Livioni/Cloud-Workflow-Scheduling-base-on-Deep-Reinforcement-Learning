import random,math,argparse
import numpy as np
from numpy.random.mtrand import sample

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='default', type=str)#参数定义
parser.add_argument('--n', default=20, type=int)#节点数量
parser.add_argument('--prob', default=0.2, type=float)#跳跃概率
parser.add_argument('--max_out', default=2, type=float)#最大出口
parser.add_argument('--alpha',default=1,type=float) #shape
parser.add_argument('--beta',default=1.0,type=float)  #规则度
parser.add_argument('--jump',default=1,type=int)     #DAG 跳跃层
args = parser.parse_args()

set_dag_size = [20,30,40,50,60,70,80,90,100]         #随机DAG数量       
set_max_out = [2,3,4,5]                              #最大出节点数
set_alpha = [0.5,1.0,2.0]                            #DAG shape
set_beta = [0.0,0.5,1.0,2.0]                         #DAG 规则度
set_jump = [1,2,3]                                   #DAG 跳跃层

##############################################initialize############################################
if args.mode != 'default':
    args.n = random.sample(set_dag_size,1)[0]
    args.max_out = random.sample(set_max_out,1)[0]
    args.alpha = random.sample(set_alpha,1)[0]
    args.beta = random.sample(set_alpha,1)[0]

length = math.floor(math.sqrt(args.n)/args.alpha)
mean_value = args.n/length
random_num = np.random.normal(loc = mean_value, scale = args.beta,  size = (length,1))    #生成随机正态分布数
###############################################division#############################################

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
for i in range(length):
    dag_list_update.append(list(range(dag_num,dag_num+len(dag_list[i]))))
    dag_num += len(dag_list_update[i])

############################################link#####################################################
into_degree = [0]*args.n#节点入度列表
out_degree = [0]*args.n#节点出度列表
edges = []#存储边的列表
pred = 0

for i in range(length-1):
    sample_list = list(range(len(dag_list_update[i+1])))
    for j in range(len(dag_list_update[i])):
        od = random.randrange(1,args.max_out+1,1)
        bridge = random.sample(sample_list,od)
        for k in bridge:
            edges.append((dag_list_update[i][j],dag_list_update[i+1][k]))#连边
            into_degree[pred+len(dag_list_update[i])+k]+=1
            out_degree[pred+j]+=1 
    pred += len(dag_list_update[i])


######################################create start node and exit node################################
for node,id in enumerate(into_degree):#给所有没有入边的节点添加入口节点作父亲
    if id ==0:
        edges.append((0,node+1))
        into_degree[node]+=1

for node,od in enumerate(out_degree):#给所有没有出边的节点添加出口节点作儿子
    if od ==0:
        edges.append((node+1,args.n+1))
        out_degree[node]+=1


print(edges)
print(into_degree,out_degree)
