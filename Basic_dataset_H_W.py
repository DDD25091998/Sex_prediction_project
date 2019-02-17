#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 14:22:55 2019

@author: danielagorduza
"""


import pandas
from sklearn import preprocessing,  svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import tree
import numpy as np

file = pandas.read_csv('weight-height.csv')






x = file[['Height','Weight']]
y = file[['Gender']]

#Create a dict with values 

gender_dict = {'Male': 1,'Female': 2}

y.Gender = [gender_dict[item] for item in y.Gender]

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size = 0.2 )


mod = LinearRegression()



mod.fit(xtrain,ytrain)

clf1 = mod.fit(xtrain,ytrain)

acc_mod = mod.score(xtest,ytest)

print('accmod', acc_mod)

#Accuracy is very low ( 60-68%) lets try with a tree 

mod2 = tree.DecisionTreeClassifier()

mod2.fit(xtrain,ytrain)
clf2 = mod2.fit(xtrain,ytrain)



acc_tree = mod2.score(xtest,ytest)


print('acctree', acc_tree)


print("reg pred", clf1.predict([[79,221]]))
print("Tree pred",clf2.predict([[79,221]]))


#open(‘weight-height.csv’, ‘rb’) as f:
#reader = csv.reader(f)