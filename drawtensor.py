import csv
import matplotlib.pyplot as plt
import pandas as pd

df1 = pd.read_csv('/home/azhihong/JL-DCF-pytorch/log/run16-07/07-16-training loss.csv')  # csv文件所在路径
step1 = df1['Step'].values.tolist()
loss1 = df1['Value'].values.tolist()

df2 = pd.read_csv('/home/azhihong/JL-DCF-pytorch/log/run17-08/08-17-training loss.csv')
step2 = df2['Step'].values.tolist()
loss2 = df2['Value'].values.tolist()

plt.plot(step1, loss1, label='JL_DCF')
plt.plot(step2, loss2, label='Ours')
plt.legend(fontsize=16)  # 图注的大小
plt.show()