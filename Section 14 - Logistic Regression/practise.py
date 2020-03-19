#Logistic Regression

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset

dataset=pd.read_csv('Social_Network_Ads.csv')
X=dataset.iloc[ : , [2 , 3]].values
Y=dataset.iloc[ : , 4].values

# removing the missing values

from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan , strategy='mean')
X=imp.fit_transform(X)

#Splitting the test and train data

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

# Fitting the classifier
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)

#Prediction of the model

y_pred=classifier.predict(X_test)

#Confusion Matrix

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred)