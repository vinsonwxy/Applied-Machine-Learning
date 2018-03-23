import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('/Users/Vinson/desktop/iris.data.csv', header = None)
data = df.values

colors = []
for i in range(149):
	if data[i, 4] == 'Iris-setosa':
		colors.append('r')
	elif data[i, 4] == 'Iris-versicolor':
		colors.append('g')
	else:
		colors.append('b')

labels = ['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm']

for i in range(3):
	for j in range(i + 1, 4):

		xs = data[:, i]
		ys = data[:, j]

		plt.figure()
		plt.scatter(xs, ys, c = colors)
		plt.xlabel(labels[i])
		plt.ylabel(labels[j])
		
		plt.savefig('/Users/vinson/desktop/plot' + str(i) + str(j) +'.png')