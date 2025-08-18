#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 15:45:07 2025

@author: gouthamv
"""

import numpy as np
import pandas as pd
import matplotlib as plt


dataset = pd.read_csv(r"/Users/gouthamv/GitHub/Machine_Learning_ML/ml_data.csv")
print(dataset)

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
imputer = imputer.fit(X[:, 1:3])

X[:, 1:3] = imputer.transform(X[:, 1:3])

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()

X[:, 0] = labelencoder_X.fit_transform(X[:,0])

labelEncoder_Y = LabelEncoder()

Y = labelEncoder_Y.fit_transform(Y)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=0)





