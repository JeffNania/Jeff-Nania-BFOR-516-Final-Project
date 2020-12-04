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



#%% Create labels for binary classification/Convert digital binary to textual binary

Diabetes.Outcome.replace({0:'No Diabetes',1:'Positive for Diabetes'},inplace=True)

print(Diabetes.Outcome.value_counts())

#%% Segment Data

train, test = train_test_split(Diabetes, test_size=0.25)

print("Rows in train:", len(train))

print("Rows in test:", len(test))

#%% Train Model

#Decision Tree

# define new tree
dt = tree.DecisionTreeClassifier()

# train the model using the 2nd and 6th columns (Glucose and BMI)
# The value we are trying to predict is 'Outcome'

dt.fit(train.iloc[:, 2:6], train['Outcome'])

# Tree depth
print(dt.get_depth())


#%% Predict Labels for Test Data (Using Decision Tree)

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
#%% Random Forest 

rf = RandomForestClassifier()

rf.fit(train[Diabetes_Predictors], train['Outcome'])

#%% Predict Labels for Test Data (Using Random Forest)

predicted = rf.predict(test[Diabetes_Predictors])

print(predicted[:5]) # show first five predictions

# count test data
test_labels_stats = Counter(test['Outcome'])

print("Labels in the test data:", test_labels_stats)

# count predicted
predicted_labels_stats = Counter(predicted)

print("Labels in the predictions:", predicted_labels_stats)

#%% Naive Bayes 

nb = GaussianNB()

nb.fit(train[Diabetes_Predictors], train['Outcome'])

#%% Predict Labels for Test Data (Using Gaussian Naive Bayes)

predicted = nb.predict(test[Diabetes_Predictors])

print(predicted[:5]) # show first five predictions

# count test data
test_labels_stats = Counter(test['Outcome'])

print("Labels in the test data:", test_labels_stats)

# count predicted
predicted_labels_stats = Counter(predicted)

print("Labels in the predictions:", predicted_labels_stats)

#%% Logistic Regression

lr = LogisticRegression()

lr.fit(train[Diabetes_Predictors], train['Outcome'])

#%% Predict Labels for Test Data (Using Logistic Regression)

predicted = lr.predict(test[Diabetes_Predictors])

print(predicted[:5]) # show first five predictions

# count test data
test_labels_stats = Counter(test['Outcome'])

print("Labels in the test data:", test_labels_stats)

# count predicted
predicted_labels_stats = Counter(predicted)

print("Labels in the predictions:", predicted_labels_stats)

#%% Evaluation # Fix this

# list of our models
fitted = [dt, rf, nb, lr]

# empty dataframe to store the results
Model_Evaluation = pd.DataFrame(columns=['classifier_name', 'fpr','tpr','auc', 
                                     'log_loss', 'clf_report'])

for clf in fitted:
    # print the name of the classifier
    print(clf.__class__.__name__)
    
    # get predictions
    yproba = clf.predict_proba(test[Diabetes_Predictors])
    yclass = clf.predict(test[Diabetes_Predictors])
 
    """why doesn't this part work below???"""   
    
   """ #Convert textual binary back to digital binary
    Diabetes.Outcome.replace({'No Diabetes':0,'Positive for Diabetes':1},inplace=True)"""

    
    # auc information
    fpr, tpr, _ = metrics.roc_curve(test['Outcome'],  yproba[:,1])
    auc = metrics.roc_auc_score(test['Outcome'], yproba[:,1])
    
    # log loss
    log_loss = metrics.log_loss(test['Outcome'], yproba[:,1])
    print(log_loss)
    # add some other stats based on confusion matrix
    clf_report = metrics.classification_report(test['Outcome'], yclass)
    print(clf_report)    
    # add the results to the dataframe
    result_table = result_table.append({'classifier_name':clf.__class__.__name__,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc,
                                        'log_loss': log_loss,
                                        'clf_report': clf_report}, ignore_index=True)

###Below should be easy read out but also not working###
result_table.set_index('classifier_name', inplace=True)
display(result_table)

#%% View The Results





#%% Questions/Answers Available on separate PDF