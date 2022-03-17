import gym,xlwt
import numpy as np
from itertools import count

def initial_excel():
    global worksheet, workbook
    # xlwt 库将数据导入Excel并设置默认字符编码为ascii
    workbook = xlwt.Workbook(encoding='ascii')
    # 添加一个表 参数为表名
    worksheet = workbook.add_sheet('resources usage')
    # 生成单元格样式的方法
    # 设置列宽, 3为列的数目, 12为列的宽度, 256为固定值
    for i in range(3):
        worksheet.col(i).width = 256 * 12
    # 设置单元格行高, 25为行高, 20为固定值
    worksheet.row(1).height_mismatch = True
    worksheet.row(1).height = 20 * 25
    worksheet.write(0, 0, 'time')
    worksheet.write(0, 1, 'CPU usage(%)')
    worksheet.write(0, 2, 'Memory usage(%)')
    for i in range(3):
        worksheet.write(1, i, 0)
    # 保存excel文件
    workbook.save('data/tetrisres_monitor.xls')    

def check_res(state):
    job_cpu_demand = state[33:63]
    job_memory_demand = state[63:93]
    cpu_res = state[1]
    memory_res = state[2]
    for i in range(len(job_cpu_demand)):
        if ((job_cpu_demand[i] == -1.0) and (job_memory_demand[i] == -1.0)):
            continue
        else: 
            if (job_cpu_demand[i] > cpu_res or job_memory_demand[i] > memory_res):
                job_cpu_demand[i] = -1.0
                job_memory_demand[i] = -1.0
            else:
                continue  
    state[33:63] = job_cpu_demand
    state[63:93] = job_memory_demand
    return np.array(state, dtype=np.float32)

def alignment_score(state):
    job_cpu_demand = state[33:63]
    job_memory_demand = state[63:93]
    cpu_res = state[1]
    memory_res = state[2]
    alignment_score = cpu_res * job_cpu_demand + memory_res * job_memory_demand
    if all(map(lambda x : x<0,alignment_score)):
        return -1
    else:
        return np.where(alignment_score == np.max(alignment_score))[0][0]

initial_excel()
env = gym.make("testEnv-v0").unwrapped
print("Tetris")
line = 1 
state = env.reset()
sum_reward = 0      #记录每一幕的reward
for i in count():
    valid_state = check_res(state)
    action = alignment_score(valid_state)
    if action == -1:
        time, cpu_usage, memory_usage = env.return_res_usage()

        worksheet.write(line, 1, str(100-cpu_usage)+'%')
        worksheet.write(line, 2, str(100-memory_usage)+'%')            
        line += 1
    next_state,reward,done,info = env.step(action)
    if action == -1:
        time, cpu_usage, memory_usage = env.return_res_usage()
        worksheet.write(line-1, 0, time)
    sum_reward += reward
    #记录资源使用率

    state = next_state
    if done:
        break

workbook.save('data/tetrisres_monitor.xls')
env.close()