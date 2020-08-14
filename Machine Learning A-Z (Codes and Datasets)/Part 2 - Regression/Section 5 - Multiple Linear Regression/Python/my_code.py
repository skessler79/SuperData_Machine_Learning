#
# Multiple Linear Regression
#

### Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### Importing the dataset

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

### Encoding categorical data

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

# You don't need to remove a dummy variable because the library automatically does it
# but I will drop it on my own since it is good practice

X = X[:, 1:]

print(X)

### Splitting the dataset into Training Set and Test Set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

### Training the Multiple Linear Regression model on the Training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

### Predicting the Test set results

y_pred = regressor.predict(X_test)
np.set_printoptions(precision = 2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

### Making a single prediction
# For example a startup with R&D Spend = 160,000 ,
#                            Administration spend = 130,000 ,
#                            Marketing Spend = 300,000 ,
#                            State = 'California'

print(regressor.predict([[0, 0, 160000, 130000, 300000]]))

### Getting the final linear regression equation with the values of the coefficients

print(regressor.coef_)
print(regressor.intercept_)






























