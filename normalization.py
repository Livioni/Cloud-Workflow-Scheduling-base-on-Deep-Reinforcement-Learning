datas = [347.19, 370.16, 363.57, 370.48, 377.65, 422.66]
norm_data = []
for i in range(len(datas)):
    norm_data.append(datas[i] / max(datas))
print(norm_data)
