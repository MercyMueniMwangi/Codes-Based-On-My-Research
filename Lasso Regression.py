# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 11:27:06 2020

@author: Mercy Mueni Mwangi
"""

#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

#import data
dataset = pd.read_csv("C:\\Users\\Mercy Mueni Mwangi\\Documents\\Data Science\\Udemy\\Weight.csv")
dataset.head()

#reshaping data set to 2d array
Y = dataset["Weight"].values.reshape(-1,1)
X = dataset["Height"].values.reshape(-1,1)
#splitting data
trainx,testx,trainy,testy=train_test_split(X,Y,test_size = 0.7,random_state = 0)

#model
regressor=LinearRegression()
regressor.fit(trainx,trainy)

#plot the tain data
plt.scatter(trainx,trainy, color="red")
plt.plot(trainx,regressor.predict(trainx),color="black")
plt.title("Weight Regressor Plot",loc="center")
plt.ylabel('Weight in kg')
plt.xlabel("Height in cm")
plt.show()

#scoring the model using train data
ytrain_predict=regressor.predict(trainx)
scoretraindata = r2_score(ytrain_predict,trainy)

#ploting the test data
plt.scatter(testx,testy, color="red")
plt.plot(testx,regressor.predict(testx),color="black")
plt.title("Weight Regressor Plot",loc="center")
plt.ylabel('Weight in kg')
plt.xlabel("Height in cm")
plt.show()

#score the model against the test data
ytest_predict=regressor.predict(testx)
scoretestdata = r2_score(ytest_predict,testy)

#Implementing Lasso Regression
Lassoalpha = Lasso(alpha=3)
Lassoalpha.fit(trainx,trainy)

##ploting the train data after adding a bias
plt.scatter(trainx,trainy, color="red")
plt.plot(trainx,Lassoalpha.predict(trainx),color="black")
plt.title("Weight Regressor Plot",loc="center")
plt.ylabel('Weight in kg')
plt.xlabel("Height in cm")
plt.show()

#score the ridge regression with train data
ytrain_predict_Lasso=Lassoalpha.predict(trainx)
scoretraindataLasso = r2_score(trainy,ytrain_predict_Lasso)

##ploting the test data after adding a bias
plt.scatter(testx,testy, color="red")
plt.plot(testx,Lassoalpha.predict(testx),color="black")
plt.title("Weight Regressor Plot",loc="center")
plt.ylabel('Weight in kg')
plt.xlabel("Height in cm")
plt.show()

#score the ridge regression with train data
ytest_predict_Lasso=Lassoalpha.predict(testx)
scoretestdataLasso = r2_score(testy,ytest_predict_Lasso)

#to predict a value
while True:
    try:
        value=input("Enter the value of your height")
        Value= np.array([[float(value)]])
        Weight= Lassoalpha.predict(Value)
        print('The weight asscociated with the height %s cm is' %(value), float(Weight) , 'Kg' )
    except ValueError:
        print('Oops! Not a number!')
        print('Please enter a number')
    else:
        break


