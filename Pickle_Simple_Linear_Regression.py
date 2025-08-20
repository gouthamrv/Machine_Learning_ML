#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 11:55:31 2025

@author: gouthamv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
dataset = pd.read_csv(
    r'/Users/gouthamv/GitHub/Machine_Learning_ML/Salary_Data.csv')

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=0)

x_train = x_train.values.reshape(-1, 1)
x_test = x_test.values.reshape(-1, 1)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)


plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Predict Salary for 12 and 20 years of experience
y_12 = regressor.predict([[12]])
y_20 = regressor.predict([[20]])
print(f"Predicted Salary for 12 years of experience: ${y_12[0]:,.2f}")
print(f"Predicted Salary for 20 years of experience: ${y_20[0]:,.2f}")

# Save the model to Disk using pickle
import pickle
with open('simple_linear_regression_model.pkl', 'wb') as file:
    pickle.dump(regressor, file)

'''
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Traing set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

coef = print(f"Coeff: {regressor.coef_}")
intercept = print(f"Intercept: {regressor.intercept_}")

comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)

#====PREDICTION TEST========

salary_of_12_yrs_exp = 9312 * 12 + 26780 #y^ = mx+c
print(salary_of_12_yrs_exp)

bias = regressor.score(x_train, y_train)
variance = regressor.score(x_test, y_test)
print(bias, variance)

#inferentail stats
#z-score
dataset.apply(stats.zscore)

#sum of square regression (SSR)
y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

#SST
print(dataset.values)
mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values-mean_total)**2)
print(SST)

#R square
r_square = 1- (SSR/SST)
print(r_square)
'''
