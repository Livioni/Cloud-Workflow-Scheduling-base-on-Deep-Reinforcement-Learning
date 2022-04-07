datas = [275.91, 297.74, 291.31, 297.68, 302.44, 343.35]
norm_data = []
for i in range(len(datas)):
    norm_data.append(datas[i] / max(datas))
print(norm_data)
