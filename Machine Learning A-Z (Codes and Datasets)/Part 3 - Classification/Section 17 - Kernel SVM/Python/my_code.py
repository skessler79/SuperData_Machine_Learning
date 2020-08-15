# =============================================================================
# Kernel SVM
# =============================================================================

### Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### Importing the dataset

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

### Splitting the dataset into Training Set and Test Set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

### Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

### Training the Kernel SVM on the Training Set

from sklearn.svm import SVC












