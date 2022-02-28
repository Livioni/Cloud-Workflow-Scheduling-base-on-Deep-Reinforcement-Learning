# Cloud Workflow Scheduling base on Deep Reinforcement Learning
 北京化工大学本科毕业设计《基于深度强化学习的云工作流调度》

# 基于深度强化学习的云工作流调度

## 有向无环图生成设计

  	工作流通常由DAG（有向无环图）来定义，其中每个计算任务$T_i$由一个顶点表示。同时，任务之间的每个数据或控制依赖性由一条加权的有向边$E_{ij}$表示。每个有向边$E_{ij}$表示$T_i$是$T_j$的父任务，$T_j$只能在其所有父任务完成后执行。一般在所有任务之前设立一个Start虚拟节点，作为所有没有父任务节点的父节点；同理，在所有任务之后设立一个Exit虚拟节点，作为所有没有子任务节点的子节点，这两个虚拟节点都没有计算资源需求。

​	确定表示一个DAG需要三个数据，分别是是节点连接信息，各节点的父节点数，各节点的子节点数。由这三个元素可以确定一个独立的DAG。

​	例如一个10个节点的DAG：

**Edges:** [(1, 5), (1, 6), (2, 4), (2, 6), (3, 6), (4, 7), (4, 9), (5, 9), (5, 7), (6, 7), ('Start', 1), ('Start', 2), ('Start', 3), ('Start', 8), ('Start', 10), (7, 'Exit'), (8, 'Exit'), (9, 'Exit'), (10, 'Exit')]

**In_degree:** [1, 1, 1, 1, 1, 3, 3, 1, 2, 1] 

**out_degree:** [2, 2, 1, 2, 2, 1, 1, 1, 1, 1] 

表示的是下面这一张图。

![image-20220228114904174](README.assets/image-20220228114904174.png)

参数设定为：

size = [20,30,40,50,60,70,80,90]       #DAG中任务的数量

max_out = [1,2,3,4,5]            #DAG节点的最大出度

alpha = [0.5,1.0,1.5]            #控制DAG 的形状

beta = [0.0,0.5,1.0,2.0]           #控制 DAG 的规则度

具体的实现细节如下：



1. 根据公式![img](README.assets/clip_image002.png) 计算出生成DAG的层数，并计算平均每层的数量![img](README.assets/clip_image004.png).

2. 在以均值为![img](README.assets/clip_image004.png)，标准差为![img](README.assets/clip_image007.png)的正态分布中采样每层的任务数并向上取整，这样随机采样得到的总任务数可能有偏差，随机在某几层添加或者删除任务，使DAG总任务数等于![img](README.assets/clip_image009.png)。

3. 对于第一层到倒数第二层：每一个任务随机在[0, ![img](README.assets/clip_image011.png)]中采样整数![img](README.assets/clip_image013.png)，并随机连接![img](file:////Users/livion/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image013.png)个下一层的任务。

4. 最后给所有没有入边的任务添加Start作为父节点，所有没有出边的任务添加Exit任务作为子节点，至此一个随机的DAG就生成好了。
