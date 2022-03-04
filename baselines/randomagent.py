import gym,xlwt,random
from itertools import count

env = gym.make("MyEnv-v0").unwrapped
n_iters=100
state_size = env.observation_space.shape[0] 
action_size = env.action_space.n 


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
    workbook.save('data/makespan_random.xls') 



def randomagent(n_iters):
    print("random")
    for iter in range(n_iters):
        state = env.reset()
        sum_reward = 0      #记录每一幕的reward
        time = 0            #记录makespan
        for i in count():
            action = random.choice(range(action_size))-1
            state,reward,done,info = env.step(action)
            while (info[0] == False):
                action = random.choice(range(action_size))-1
                state,reward,done, info = env.step(action)#输入step的都是
            next_state, reward, done, _ = state, reward, done, info
            # print(action)
            next_state,reward,done,info = env.step(action)
            # print(next_state)
            sum_reward += reward
            state = next_state
            if done:
                time = state[0]
                time_to_write = round(float(time),3)
                worksheet.write(iter, 0, time_to_write)
                workbook.save('data/makespan_random.xls') 
                # print('Episode: {}, Reward: {:.3f}, Makespan: {:.3f}s'.format(iter+1, sum_reward,time))
                break
             

if __name__ == '__main__':
    initial_excel()
    randomagent(n_iters)  