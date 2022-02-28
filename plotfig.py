import matplotlib.pyplot as plt
import numpy as np
import xlrd,torch
import ShortestJobFirst,randomagent,DRL_test

#open data
makespan_file = xlrd.open_workbook('datas.xls')
makespan_sheet = makespan_file.sheets()[0]
lie1 = [makespan_sheet.cell_value(i, 0) for i in range(1, makespan_sheet.nrows)]
lie2= [makespan_sheet.cell_value(i, 1) for i in range(1, makespan_sheet.nrows)]
lie3= [makespan_sheet.cell_value(i, 2) for i in range(1, makespan_sheet.nrows)]

lie4 = [makespan_sheet.cell_value(i, 3) for i in range(1, makespan_sheet.nrows)]
lie5= [makespan_sheet.cell_value(i, 4) for i in range(1, makespan_sheet.nrows)]
lie6= [makespan_sheet.cell_value(i, 5) for i in range(1, makespan_sheet.nrows)]

lie7 = [makespan_sheet.cell_value(i, 6) for i in range(1, makespan_sheet.nrows)]
lie8= [makespan_sheet.cell_value(i, 7) for i in range(1, makespan_sheet.nrows)]
lie9= [makespan_sheet.cell_value(i, 8) for i in range(1, makespan_sheet.nrows)]

all_data1 = [np.array(lie1),np.array(lie2),np.array(lie3)]
all_data2 = [np.array(lie4),np.array(lie5),np.array(lie6)]
all_data3 = [np.array(lie7),np.array(lie8),np.array(lie9)]

labels = ['DRL','SJF','random']

fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(9, 4))

# rectangular box plot
bplot1 = ax1.boxplot(all_data1,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels)  # will be used to label x-ticks
ax1.set_title('Makespan (size = 10)')

bplot2 = ax2.boxplot(all_data2,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels)  # will be used to label x-ticks
ax2.set_title('Makespan (size = 30)')

bplot3 = ax3.boxplot(all_data3,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels)  # will be used to label x-ticks
ax3.set_title('Makespan (size = 50)')

# fill with colors 
# colors = ['pink', 'lightblue', 'lightgreen']
colors = [(0,80/255,179/255),(34/255, 120/255, 4/255),(254/255, 77/255, 79/255)]
for patch, color in zip(bplot1['boxes'], colors):
    patch.set_facecolor(color)

for patch, color in zip(bplot2['boxes'], colors):
    patch.set_facecolor(color)

for patch, color in zip(bplot3['boxes'], colors):
    patch.set_facecolor(color)

# adding horizontal grid lines
for ax in [ax1]:
    ax.yaxis.grid(True)
    ax.set_xlabel('n = 10')
    ax.set_ylabel('Makespan(s)')

for ax in [ax2]:
    ax.yaxis.grid(True)
    ax.set_xlabel('n = 30')
    ax.set_ylabel('Makespan(s)')

for ax in [ax3]:
    ax.yaxis.grid(True)
    ax.set_xlabel('n = 50')
    ax.set_ylabel('Makespan(s)')


plt.show()