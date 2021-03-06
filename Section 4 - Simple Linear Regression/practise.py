#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, 1].values

#splitting the datasets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train ,Y_test=train_test_split(X,Y,test_size=1/3, random_state=0)

# Simple Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

#Predicting the testset results

y_pred=regressor.predict(X_test)

#Visualisation of training set results

plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience Plot')
plt.xlabel('Exp')
plt.ylabel('Salary')
plt.show()

# Visualisation of test set result

plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience Plot')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

