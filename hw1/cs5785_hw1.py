# cs5785
# HW1
# Q1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
from scipy.spatial import distance
from sklearn import datasets, cross_validation, svm, linear_model
from sklearn.model_selection import train_test_split, cross_val_score

train = pd.read_csv("train.csv").values
test = pd.read_csv("test.csv").values
'''
#(b)
def show_image(n):
	plt.matshow(test[n].reshape(28, 28), cmap = 'gray')
	plt.show()

interact(show_image, n=(0,42000))
#show_image(3)

#(c)
plt.figure()
plt.hist(train[:, 0], bins = 10, normed = True)

#(d)
dist_matrix = distance.cdist(train[:, 1:], train[:, 1:])

#(e)
binary = train[np.where(train[:, 0] < 2)]
zeros = binary[np.where(binary[:, 0] == 0)]
ones = binary[np.where(binary[:, 0] == 1)]

genuine = np.append(distance.pdist(zeros), distance.pdist(ones))
impostor = distance.cdist(zeros, ones).flatten()

plt.figure()
plt.hist(genuine, bins = 300, normed = True, alpha = 0.5)
plt.hist(impostor, bins = 300, normed = True, alpha = 0.5)

#(f)
fpr = []
tpr = []
genuine.sort()
impostor.sort()

max_dist = max(genuine[len(genuine) - 1], impostor[len(impostor) - 1])

for i in range(0, int(max_dist), 1):
    fpr.append(100.0 * bisect.bisect_left(impostor, i)/float(len(genuine)))
    tpr.append(100.0 * bisect.bisect_left(genuine, i)/float(len(genuine)))

plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
'''

#(g)
def kNN(k, Xtrain, Ytrain, Xtest):
	Ytest = []
	for i in range(len(Xtest)):
		print(i)

		Ytest.append(predict(k, Xtrain, Ytrain, Xtest[i, :]))
	return Ytest

def predict(k, Xtrain, Ytrain, Xtest): # Xtest gets passed in one data at a time
	dist = distance.cdist(Xtrain, Xtest.reshape(1, len(Xtest)))
	index = np.argsort(dist.flatten())[:k] # 1-dimensional array
	return mode(Ytrain[index])

def mode(arr):
	return collections.Counter(arr).most_common(1)[0][0]

#(h)
def my_cross_val(method, X, y, k):
    subset_size = len(y) // 3
    acc_rates = np.zeros(3)
    y_actual = []
    y_pred = []
    for i in range(3):
    	print("Round", i)
    	X_train = np.concatenate((X[:i * subset_size],X[(i + 1) * subset_size:]), axis = 0)
    	X_test = X[i * subset_size:][:subset_size]
    	y_train = np.concatenate((y[:i * subset_size] , y[(i + 1) * subset_size:]), axis = 0)
    	y_test = y[i * subset_size:][:subset_size]
    	result = method(k, X_train, y_train, X_test)
    	y_pred = np.append(y_pred, result)
    	y_actual = np.append(y_actual, y_test)
    	mm = 0
    	for j in range(len(result)):
    		if result[j] != y_test[j]:
    			mm = mm + 1
    	acc = 1 - float(mm)/float(len(result))
    	acc_rates[i] = acc
 #  np.save("/Desktop/y_actual.npy", y_actual)
 #  np.save("/Desktop/y_pred.npy", y_pred)
    return acc_rates
    #print(acc_rates, "mean:",np.mean(acc_rates), "standard deviation:", np.std(acc_rates))

l = my_cross_val(kNN, train[:, 1:], train[:, 0], 3)
# l
# array([ 0.9665    ,  0.965     ,  0.96778571])
# np.mean(l)
# 0.9664285714285713

#(i)

#(j)
#Ytest = kNN(3, train[:, 1:], train[:, 0], test, 3)
#np.save("/Desktop/y_final.npy", Ytest)

