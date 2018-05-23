import numpy as np


cm = np.loadtxt("cost_matrix")

for i in range(len(cm)):
	count = 0
	for j in range(len(cm[i])):
		if cm[i][j] < 100:
			count+=1
			print(i, j, cm[i][j])
	print("count=", str(count))