# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 14:19:53 2020

@author: Mercy Mueni Mwangi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#reading the file
dataset = pd.read_csv('C:\\Users\\Mercy Mueni Mwangi\\Documents\\Data Science\\Udemy\\baseballplayer.csv')
dataset.head()

#specifying our variables
X = dataset['Angle'].values.reshape(-1,1)
Y = dataset['Distance'].values.reshape(-1,1)

#ploting/visualizing the data
plt.scatter(X,Y,color = "red")
plt.xlabel("Angle")
plt.ylabel("Distance")
plt.title("Distance covered by the ball", loc = "center")
plt.show()

#Drawing linear regression line
# here we use R-squared  value to draw a linear line through a non-linear data
x_train, x_test,y_train,y_test= train_test_split(X,Y, test_size =0.4, random_state=0)

# Regression Instance
regressor= LinearRegression()
regressor.fit(x_train,y_train)

# Plotting the linear regression line through the polinomial data
plt.scatter(X,Y,color = "red")
plt.plot(x_train,regressor.predict(x_train))
plt.xlabel("Angle")
plt.ylabel("Distance")
plt.title("Distance covered by the ball", loc = "center")
plt.show()

#r-squared score
pred_y = regressor.predict(x_test)
R_squared= r2_score(y_test,pred_y)


''' USING POLYNOMIAL REGRESSION MODEL '''
from sklearn.preprocessing import PolynomialFeatures

#transforming x features with PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
transformed_x= poly_reg.fit_transform(X)

#splitting the data
xtrain, xtest,ytrain,ytest= train_test_split(transformed_x,Y, test_size =0.4, random_state=0)

# fitting a regression model
regressor2=LinearRegression()
regressor2.fit(xtrain,ytrain)

#r-squared score
pred_y1=regressor2.predict(xtest)
Rsquared= r2_score(ytest,pred_y1)

