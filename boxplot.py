import matplotlib.pyplot as plt
import numpy as np
import xlrd,torch
import ShortestJobFirst,randomagent,DRL_test
from DRLagent import Actor, Critic

actor = torch.load('models/ACagent/actor.pkl')
critic = torch.load('models/ACagent/critic.pkl')
DRL_test.initial_excel()
DRL_test.test(actor, critic)  
ShortestJobFirst.initial_excel()
ShortestJobFirst.sjf(100)
randomagent.initial_excel()
randomagent.randomagent(100)

#open data
makespan_file = xlrd.open_workbook('data/makespan_AC.xls')
makespan_sheet = makespan_file.sheets()[0]
lie1 = [makespan_sheet.cell_value(i, 0) for i in range(1, makespan_sheet.nrows)]

sjf_file = xlrd.open_workbook('data/makespan_SJF.xls')
sjf_sheet = sjf_file.sheets()[0]
lie2 = [sjf_sheet.cell_value(i, 0) for i in range(1, sjf_sheet.nrows)]

random_file = xlrd.open_workbook('data/makespan_random.xls')
random_sheet = random_file.sheets()[0]
lie3 = [random_sheet.cell_value(i, 0) for i in range(1, random_sheet.nrows)]

all_data = [np.array(lie1),np.array(lie2),np.array(lie3)]

labels = ['DRL','SJF','random']

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))

# rectangular box plot
bplot1 = ax1.boxplot(all_data,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels)  # will be used to label x-ticks
ax1.set_title('Makespan')

# fill with colors 
# colors = ['pink', 'lightblue', 'lightgreen']
colors = [(0,80/255,179/255),(34/255, 120/255, 4/255),(254/255, 77/255, 79/255)]
for patch, color in zip(bplot1['boxes'], colors):
    patch.set_facecolor(color)

# adding horizontal grid lines
for ax in [ax1]:
    ax.yaxis.grid(True)
    ax.set_xlabel('Observed values')
    ax.set_ylabel('Makespan(s)')

plt.show()