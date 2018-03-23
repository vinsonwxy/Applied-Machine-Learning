# Class notes
# 09/05/2017

import numpy as np
from sklearn import datasets, cross_validation, svm, linear_model
from ipywidgets import interact

imshow?

matshow(X[25].reshape(8, 8), cmap = 'gray')

def show_image(n):
	print "This sample's label is", Y[n]
	matshow(X[n].reshape(8, 8), cmap = 'gray')

interact(show_image, n=(0,1700))

Xtrain, Xtest, Ytrain, Ytest = cross_validation.train_test_split(X, Y)

cls = linear_model.LogisticRegression()
cls.fit(Xtrain, Ytrain)

print cls.decision_function(Xtest[25])
print "Should be", Ytest[25]

(cls.predict(Xtest) == Ytest).sum()
(cls.predict(Xtest) == Ytest).mean()

cls.score(Xtest, Ytest)

for train, test in cross_validation.KFold(len(Y), n_folds = 10, shuffle = True):
	print "New fold"
	print "Train on", train
	print "Test on", test
	cls = svm.SVC(C = 1.0, gamma = 0.001)
	score = cls.fit(X[train], Y[train]).score(X[test], Y[test])
	print C, "\t", gamma, "\t", score

for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100]:
	for gamma in [0.001, 0.01, 0.1, 1.0, 10.0, 100]:
		cls = svm.SVC(C = C, gamma = gamma)
		score = cls.fit(Xtrain, Ytrain).score(Xtest, Ytest)
		print C, "\t", gamma, "\t", score


from IPython import parallel
rc = parallel.Client()
lv = rc.load_balanced_view()
print len(lv)

%%px
print "Hello world"

%%px
import time


def evaluate(fold):
	train, test = fold
	cls = svm.SVC(C = 1.0, gamma = 0.001)
	return cls.fit(X[train], Y[train]).score(X[test], Y[test])

folds = cross_validation.KFold(len(Y), n_folds = 10, shuffle = True)

%timeit lv.map_sync(evaluate, folds)


