import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
from scipy.spatial import distance
from sklearn import datasets, cross_validation, svm, linear_model
from sklearn.model_selection import train_test_split, cross_val_score

train = pd.read_csv("train.csv").values
test = pd.read_csv("test.csv").values

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


Ytest = kNN(3, train[:, 1:], train[:, 0], test)
np.save("/Desktop/y_final.npy", Ytest)