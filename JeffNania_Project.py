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
from sklearn import metrics
from sklearn import tree
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

#%% Find Averages, Min, and Max of all factors

Diabetes_Predictors_means = Diabetes.groupby('Outcome').mean()
Diabetes_Predictors_mins = Diabetes.groupby('Outcome').min()
Diabetes_Predictors_maxes = Diabetes.groupby('Outcome').max()

#%% Create labels for binary classification #FIX THIS

Diabetes['label'] = np.where(Diabetes['Outcome'] == 'Outcome', 'good', 'bad')

#%%


#%% Simple plots #FIX THIS

# Create a plot that

Diabetes_Plot=Diabetes.groupby(['Age', 'Glucose'])['Outcome'].sum().unstack().plot(logy=False, figsize=(100,10))
Diabetes_Plot.set(xlabel='Diabetes_Predictors', ylabel='Diabetes')
Diabetes_Plot.get_figure()

#%% Segment Data #FIX THIS

train, test = train_test_split(netattacks, test_size=0.25)
print("Rows in train:", len(train))
print("Rows in test:", len(test))

#%% Train Model #FIX THIS

# define new tree
dt = tree.DecisionTreeClassifier()
# train the model using the 9th and 10th columns
# The value we are trying to predict is 'label'
dt.fit(train.iloc[:, 9:10], train['label'])

#%% Questions

# 1) What factor or factors seem to be most closely linked with diabetes?
"""By using simple correlations, we can see that Glucose and BMI have the highest correlation with Diabetes.
Glucose has a score of roughly 0.47 and BMI has a score of roughly 0.29"""


# Which models seem to perform the best for this prediction?


# Do the models perform best when considering all the factors or just some of the factors?
