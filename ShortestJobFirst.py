import gym,xlwt
from itertools import count

env = gym.make("MyEnv-v0").unwrapped
n_iters=100

def initial_excel():
    global worksheet,workbook
    # xlwt 库将数据导入Excel并设置默认字符编码为ascii
    workbook = xlwt.Workbook(encoding='ascii')
    #添加一个表 参数为表名
    worksheet = workbook.add_sheet('makespan')
    # 生成单元格样式的方法
    # 设置列宽, 3为列的数目, 12为列的宽度, 256为固定值
    for i in range(3):
        worksheet.col(i).width = 256 * 12
    # 设置单元格行高, 25为行高, 20为固定值
    worksheet.row(1).height_mismatch = True
    worksheet.row(1).height = 20 * 25
    # 保存excel文件
    workbook.save('data/makespan_SJF.xls') 

def find_shortest_job(state):
    '''
    寻找shortest的job
    :param state: 当前状态 
    :return: shortest job在[0:9]中的索引
    '''
    ready_job_list = state[3:13].tolist()
    min = 999999
    for ele in ready_job_list:
        if ele != -1:
            min = ele if ele < min else min
    shortest_ind = ready_job_list.index(min)
    return shortest_ind 

def check_res(state):
    '''
    判断当前机器是否还可以装载
    :param state: 当前状态 
    :return: bool值 是否还可以装载
    '''
    job_duration = state[3:13].tolist()
    job_cpu_demand = state[13:23].tolist()
    job_memory_demand = state[23:33].tolist()
    cpu_res = state[1]
    memory_res = state[2]
    flag = False
    for i in range(len(job_duration)):
        if ((job_cpu_demand[i] == -1.0) and (job_memory_demand[i] == -1.0)):
            continue
        else: 
            flag = True if (job_cpu_demand[i] < cpu_res and job_memory_demand[i] < memory_res) else False
            if flag == True:
                break
    return flag

def check_ready(state,index):
    '''
    判断当前机器是否还可以装载任务index
    :param state: 当前状态 
    :param index: 查询的任务index 
    :return: bool值 是否还可以装载
    '''
    job_duration = state[3:13].tolist()
    job_cpu_demand = state[13:23].tolist()
    job_memory_demand = state[23:33].tolist()
    cpu_res = state[1]
    memory_res = state[2]
    return True if (job_cpu_demand[index] < cpu_res and job_memory_demand[index] < memory_res) else False


def sjf(n_iters):
    for iter in range(n_iters):
        state = env.reset()
        sum_reward = 0      #记录每一幕的reward
        time = 0            #记录makespan
        for i in count():
            if (check_res(state)):
                preaction = find_shortest_job(state)
                if check_ready(state,preaction):
                    action = preaction
                else:
                    action = -1
            else:
                action = -1
            # print(action)
            next_state,reward,done,info = env.step(action)
            # print(next_state)
            sum_reward += reward
            state = next_state
            if done:
                time = state[0]
                time_to_write = round(float(time),3)
                worksheet.write(iter, 0, time_to_write)
                workbook.save('data/makespan_SJF.xls') 
                # print('Episode: {}, Reward: {:.3f}, Makespan: {:.3f}s'.format(iter+1, sum_reward,time))
                break
             

if __name__ == '__main__':
    initial_excel()
    sjf(n_iters)  