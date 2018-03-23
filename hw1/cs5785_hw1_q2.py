# cs5785
# HW1
# Q2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

train = pd.read_csv("train2.csv").values
test = pd.read_csv("test2.csv").values

is_female = np.zeros(891)
is_female[np.where(train[:, 4] == 'female')] = 1

is_child = np.zeros(891)
is_child[np.where(train[:, 5].astype(float) < 18)] = 1

train = np.append(train, is_female.reshape(891, 1), axis = 1)
train = np.append(train, is_child.reshape(891, 1), axis = 1)

features = [2, 12, 13]
Xtrain = train[:, features].astype(int)
Ytrain = train[:, 1].astype(int)

cls = linear_model.LogisticRegression()
cls.fit(Xtrain, Ytrain)
print(cls.score(Xtrain, Ytrain))

