# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 11:45:01 2020

@author: Jeff Nania

The data for this project comes from UCI Machine Learning, but is available at Kaggle
through the following link: https://www.kaggle.com/uciml/pima-indians-diabetes-database . 
The data is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. 
The patients considered in this dataset are all females above the age of 21 of Pima Indian Heritage. 
The Pima people are a group of Native Americans who live in what is now south and central Arizona 
as well as northwestern Mexico. 

This data set considers factors such as glucose concentration, blood pressure, body mass index, age, 
and more to attempt to predict whether or not an individual has diabetes.  

"""
#%% Python Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import Counter
from sklearn import metrics
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import (KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

#%% Import Data

# Data for this project is available at https://www.kaggle.com/uciml/pima-indians-diabetes-database?select=diabetes.csv

# Read in dataset
Diabetes = pd.read_csv('diabetes.csv')

#%%

# Specify Variabless

Diabetes_Predictors = ['Age', 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']

#%% Column Summaries

print(Diabetes.describe(include='all'))

Diabetes.head()
#%% Histograms

Diabetes['Age'].hist()

Diabetes['Pregnancies'].hist()

Diabetes['Glucose'].hist()

Diabetes['BloodPressure'].hist()

Diabetes['SkinThickness'].hist()

Diabetes['Insulin'].hist()

Diabetes['BMI'].hist()

Diabetes['DiabetesPedigreeFunction'].hist()

Diabetes['Outcome'].hist()

#%% Correlations

Diabetes['Age'].corr(Diabetes['Outcome'])

Diabetes['Pregnancies'].corr(Diabetes['Outcome'])

Diabetes['Glucose'].corr(Diabetes['Outcome'])

Diabetes['BloodPressure'].corr(Diabetes['Outcome'])

Diabetes['SkinThickness'].corr(Diabetes['Outcome'])

Diabetes['Insulin'].corr(Diabetes['Outcome'])

Diabetes['BMI'].corr(Diabetes['Outcome'])

Diabetes['DiabetesPedigreeFunction'].corr(Diabetes['Outcome'])

#%% Find Averages, Min, and Max of all factors as well as count

# mean, min, max

Diabetes_Predictors_means = Diabetes.groupby('Outcome').mean()

Diabetes_Predictors_mins = Diabetes.groupby('Outcome').min()

Diabetes_Predictors_maxes = Diabetes.groupby('Outcome').max()

# Counts

Diabetes_Predictors_count = Diabetes.groupby('Outcome').count()

Diabetes_Predictors_count = Diabetes.groupby('Outcome').agg(['count','mean'])

Diabetes_Positive_count = Diabetes.groupby(Outcome==1).count() ## DOES NOT WORK


#%% Create labels for binary classification #FIX THIS

Diabetes.Outcome.replace({0:'No Diabetes',1:'Positive for Diabetes'},inplace=True)

print(Diabetes.Outcome.value_counts())

#%% Segment Data #FIX THIS

train, test = train_test_split(Diabetes, test_size=0.25)

print("Rows in train:", len(train))

print("Rows in test:", len(test))

#%% Train Model #FIX THIS

#Decision Tree

# define new tree
dt = tree.DecisionTreeClassifier()

# train the model using the 2nd and 6th columns (Glucose and BMI)
# The value we are trying to predict is 'Outcome'

dt.fit(train.iloc[:, 2:6], train['Outcome'])

# Tree depth
print(dt.get_depth())


#%% Predict Labels for Test Data # FIX THIS

predicted = dt.predict(test.iloc[:, 2:6])

print(predicted[:5]) # show first five predictions

# count test data
test_labels_stats = Counter(test['Outcome'])

print("Labels in the test data:", test_labels_stats)

# count predicted
predicted_labels_stats = Counter(predicted)

print("Labels in the predictions:", predicted_labels_stats)
#%% Model Stats # FIX THIS

# Classification report

print(metrics.classification_report(test['Outcome'], Diabetes_Predictors, digits=5))

#Confusion Matrix

metrics.confusion_matrix(y_true=test['label'], y_pred=predicted, labels=['good', 'bad'])

metrics.plot_confusion_matrix(dt, test.iloc[:,2:6], test['label'], labels=['good', 'bad'])

#Accuracy

# compute baseline accuracy (predict all bad)

baseline = test_labels_stats['bad'] / (test_labels_stats['good'] + test_labels_stats['bad'])

print("Baseline accuracy is:", baseline)

# compute the observed accuracy

acc = metrics.accuracy_score(test['label'], predicted)

print("Observed accuracy is:", acc)

result = metrics.classification_report(test['label'], predicted, digits=4)

print(result)
#%% Random Forest # Fix This

rf = ensemble.RandomForestClassifier()

rf.fit(train[pred_vars], train['DEATH_EVENT'])

#%% Naive Bayes # Fix This

nb = GaussianNB()

nb.fit(train[pred_vars], train['DEATH_EVENT'])

#%% Logistic Regression # Fix This

lr = LogisticRegression()

lr.fit(train[pred_vars], train['DEATH_EVENT'])

#%% Evaluation




#%% View The Results





#%% Questions

# 1) What factor or factors seem to be most closely linked with diabetes?
"""By using simple correlations, we can see that Glucose and BMI have the highest correlation with Diabetes.
Glucose has a score of roughly 0.47 and BMI has a score of roughly 0.29"""


# 2) Which models seem to perform the best for this prediction?


# Do the models perform best when considering all the factors or just some of the factors?
