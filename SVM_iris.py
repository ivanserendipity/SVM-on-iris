#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 19:24:50 2017

@author: Serendipity
"""
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn import metrics

iris = datasets.load_iris()

X = iris.data

Y = iris.target

Y = Y.reshape(150,1)

data = np.hstack((X,Y))

train,test = train_test_split(data,test_size = 0.1)

train_X = train[:,0:4]
train_Y = train[:,4]

test_X = test[:,0:4]
test_Y = test[:,4]

model = svm.SVC()
model.fit(train_X,train_Y)

prediction = model.predict(test_X)

accuracy = metrics.accuracy_score(prediction,test_Y)

print 'The accuracy is %f' %accuracy