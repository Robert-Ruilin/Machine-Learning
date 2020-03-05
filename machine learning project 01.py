# -*- coding: utf-8 -*-
"""
Machine Learning Project 01
"""
import scipy as sy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# Load dataset
url = "http://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names = names)
# Summarize the dataset
dataset_dim = dataset.shape #the number of instance and attribute in dataset
dataset_f20 = dataset.head(20) #the first 20 rows of the data
dataset_stat = dataset.describe() #the statistical summary of each attribute
dataset_class = dataset.groupby('class').size() #the number of instances that belong to each class
# Data visualization
dataset.plot(kind = 'box', subplots = True, layout = (2,2), sharex = False, sharey = False)
plt.show() #create box and whisker plots of each variable
dataset.hist()
plt.show() #create a histogram of each variable to get the distribution
pd.plotting.scatter_matrix(dataset)
plt.show() #create scatterplots of all pairs of attributes, which is helpful to spot structured relatinoships between variables
# Split the loaded dataset into 2, 80% of which we will use to train, and 20% that we will hold back as a validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size = 0.20, random_state = 1)
# Algorithm model set
models = []
models.append(('LR', LogisticRegression(solver = 'liblinear', multi_class = 'ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma = 'auto')))
# Evaluate each model
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits = 10, random_state = 1, shuffle = True) #split dataset into 10 parts, train on 9 and test on 1 and repeat for all combinations of train-test splits
    cv_results = cross_val_score(model, X_train, Y_train, cv = kfold, scoring = 'accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
plt.boxplot(results, labels = names)
plt.title('Model Comparison')
plt.show()
# Evaluate predictions
model = SVC(gamma = 'auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

   
    

