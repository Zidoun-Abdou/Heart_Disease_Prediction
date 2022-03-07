# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 20:46:43 2022

@author: MD
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

data = pd.read_csv('../input/heart-disease-uci/heart.csv')
data.info()

#Preprocessing
def onehot_encode(df, column_dict):
    df = df.copy()
    for column, prefix in column_dict.items():
        dummies = pd.get_dummies(df[column], prefix=prefix)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(column, axis=1)
    return df


def preprocess_inputs(df, scaler):
    df = df.copy()
    
    # One-hot encode the nominal features
    nominal_features = ['cp', 'slope', 'thal']
    df = onehot_encode(df, dict(zip(nominal_features, ['CP', 'SL', 'TH'])))
    
    # Split df into X and y
    y = df['target'].copy()
    X = df.drop('target', axis=1).copy()
    
    # Scale X
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X, y

X, y = preprocess_inputs(data, MinMaxScaler())

#Training
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

#Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
print("Logistic Regression Accuracy: {:.2f}%".format(lr_model.score(X_test, y_test) * 100))

#SVM
svm_model = SVC()
svm_model.fit(X_train, y_train)
print("Support Vector Machine Accuracy: {:.2f}%".format(svm_model.score(X_test, y_test) * 100))

#Neural Network
nn_model = MLPClassifier()
nn_model.fit(X_train, y_train)
print("Neural Network Accuracy: {:.2f}%".format(nn_model.score(X_test, y_test) * 100))



