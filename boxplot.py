import matplotlib.pyplot as plt
import numpy as np
import xlrd
#open data
makespan_file = xlrd.open_workbook('makespan.xls')
makespan_sheet = makespan_file.sheets()[0]
lie = [makespan_sheet.cell_value(i, 0) for i in range(1, makespan_sheet.nrows)]

all_data = np.array(lie)

labels = ['x1']

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))

# rectangular box plot
bplot1 = ax1.boxplot(all_data,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels)  # will be used to label x-ticks
ax1.set_title('Makespan')


# fill with colors 
# colors = ['pink', 'lightblue', 'lightgreen']
colors = [(0,80/255,179/255)]
for patch, color in zip(bplot1['boxes'], colors):
    patch.set_facecolor(color)

# adding horizontal grid lines
for ax in [ax1]:
    ax.yaxis.grid(True)
    ax.set_xlabel('Three separate samples')
    ax.set_ylabel('Observed values')

plt.show()