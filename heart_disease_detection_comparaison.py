# -*- coding: utf-8 -*-
"""
Created on Mon Mar  26 20:46:43 2022

@author: MD
"""

import numpy as np
import pandas as pd
import tensorflow as tf                

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neural_network import MLPClassifier


import seaborn as sns
sns.set_style('darkgrid')


tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.mode.chained_assignment = None
np.random.seed(100)

myData = pd.read_csv("heart.csv")
myData.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved', 'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']

myData['sex'][myData['sex'] == 0] = 'female'
myData['sex'][myData['sex'] == 1] = 'male'
myData['chest_pain_type'][myData['chest_pain_type'] == 1] = 'typical angina'
myData['chest_pain_type'][myData['chest_pain_type'] == 2] = 'atypical angina'
myData['chest_pain_type'][myData['chest_pain_type'] == 3] = 'non-anginal pain'
myData['chest_pain_type'][myData['chest_pain_type'] == 4] = 'asymptomatic'
myData['fasting_blood_sugar'][myData['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
myData['fasting_blood_sugar'][myData['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'
myData['rest_ecg'][myData['rest_ecg'] == 0] = 'normal'
myData['rest_ecg'][myData['rest_ecg'] == 1] = 'ST-T wave abnormality'
myData['rest_ecg'][myData['rest_ecg'] == 2] = 'left ventricular hypertrophy'
myData['exercise_induced_angina'][myData['exercise_induced_angina'] == 0] = 'no'
myData['exercise_induced_angina'][myData['exercise_induced_angina'] == 1] = 'yes'
myData['st_slope'][myData['st_slope'] == 1] = 'upsloping'
myData['st_slope'][myData['st_slope'] == 2] = 'flat'
myData['st_slope'][myData['st_slope'] == 3] = 'downsloping'
myData['thalassemia'][myData['thalassemia'] == 1] = 'normal'
myData['thalassemia'][myData['thalassemia'] == 2] = 'fixed defect'
myData['thalassemia'][myData['thalassemia'] == 3] = 'reversible defect'

myData = pd.get_dummies(myData, drop_first=True)

# Categories have been spread out in a 'one-hot' encoding
# Age, blood pressure, cholesterol, etc are still numbers, not one-hot
# We're going to want to normalize these categories to either [0,1] or [-1,1]


myData = (myData - np.min(myData)) / (np.max(myData) - np.min(myData))
x_train, x_test, y_train, y_test = train_test_split(myData.drop('target', axis=1),
                                                    myData['target'], test_size=.2, random_state=2)


lin_model = LogisticRegression(solver='lbfgs')
lin_model.fit(x_train, y_train)
print("Linear Model Accuracy: ", lin_model.score(x_test, y_test))

knn_model = KNeighborsClassifier()
knn_model.fit(x_train, y_train)
print("K Nearest Neighbor Model Accuracy: ", knn_model.score(x_test, y_test))

svm_model = SVC(gamma='auto')
svm_model.fit(x_train, y_train)
print("Support Vector Machine Model Accuracy: ", svm_model.score(x_test, y_test))

nb_model = GaussianNB()
nb_model.fit(x_train, y_train)
print("Naive Bayes Model Accuracy: ", nb_model.score(x_test, y_test))

tree_model = DecisionTreeClassifier()
tree_model.fit(x_train, y_train)
print("Decision Tree Model Accuracy: ", tree_model.score(x_test, y_test))

forest_model = RandomForestClassifier(n_estimators=100)
forest_model.fit(x_train, y_train)
print("Random Forest Model Accuracy: ", forest_model.score(x_test, y_test))


# neural networks part

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

X, y = preprocess_inputs(myData, MinMaxScaler())

#Training
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)


#Neural Network
nn_model = MLPClassifier()
nn_model.fit(X_train, y_train)
print("Neural Network Accuracy: {:.2f}%".format(nn_model.score(X_test, y_test) * 100))
