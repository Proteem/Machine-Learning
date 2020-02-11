#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importting the dataset

dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[: , :-1]
Y= dataset.iloc[:, 4]

# Categorical Dataframe

from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X=LabelEncoder()
X.iloc[:, 3]=labelencoder_X.fit_transform(X.iloc[:, 3])
transformer = ColumnTransformer([("State", OneHotEncoder(), [3])],remainder='passthrough')
X = np.array(transformer.fit_transform(X), dtype=np.float)

# Avoiding Dummy Trap
X= X[:, 1:]

# Splitting the test data and the train data
from sklearn.model_selection import train_test_split as tts
X_train,X_test,Y_train,Y_test= tts(X,Y, test_size=0.2,random_state=0)

#Fitting Multiple Linear Regression

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

# Predicting the test result

y_pred=regressor.predict(X_test)