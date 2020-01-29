#Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
dataset = pd.read_csv('C:\\Users\\Mercy Mueni Mwangi\\Documents\\Data Science\\Udemy\\billdiscount.csv')
dataset.head()
X = dataset['Bill'].values.reshape(-1,1)
Y = dataset['Discount'].values.reshape(-1,1)
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.4,random_state = 0)
#ploting the Dataset
plt.scatter(X,Y, color="red")
plt.title("Bill vs Discount")
plt.ylabel("Discount")
plt.xlabel("Bill")
plt.show()

#Creating a model
regressor= LinearRegression()
regressor.fit(x_train,y_train)

#plotting the predicted line on the train data
predict= regressor.predict(x_train)
plt.plot(x_train,predict, color="purple")
plt.scatter(x_train,y_train,color="red")
plt.title("Bill vs Discount")
plt.ylabel("Discount")
plt.xlabel("Bill")
plt.show()

##plotting the predicted line on the test data
predict= regressor.predict(x_train)
plt.plot(x_train,predict, color="purple")
plt.scatter(x_test,y_test,Y, color="red")
plt.title("Bill vs Discount")
plt.ylabel("Discount")
plt.xlabel("Bill")
plt.show()

# Scoring the train dataset
y_train_pred = regressor.predict(x_train)
r_squared_train = r2_score(y_train,y_train_pred)

# Scoring the test dataset
y_test_pred = regressor.predict(x_test)
r_squared_test = r2_score(y_test,y_test_pred)

#great r squared score

# Creating a function that returns the users discount
def Discount(bill):
    bill_int = float(bill)
    transformed_bill = np.array([[bill_int]])
    discount = int(regressor.predict(transformed_bill))
    return "Your Dicount for  %s bill amount is: %s" %(bill,discount)

# lets predict a dicount for a bill of 200
user_bill = input("Please Enter Your Bill: ")
Discount(user_bill)
    
