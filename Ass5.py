# ASSIGNMENT-5
# Classify the email using the binary classification method. Email Spam detection has two states: 

import pandas as pd
import numpy as np

df= pd.read_csv('emails.csv')

df.dtypes

df.head()

df.info()

# Check for null values
df.isnull().values.any()

df.describe()

import seaborn as sns
import matplotlib.pyplot as plt

x = df.iloc[:,1:3001]
x.shape
x

y = df.iloc[:,-1].values
y.shape
y

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 1)
x_train.shape, x_test.shape

print("The shape of X_train is:",x_train.shape)

print("The shape of X_test is:",x_test.shape)

print("The shape of y_train is:",y_train.shape)

print("The shape of y_test is:",y_test.shape)

from sklearn.metrics import accuracy_score
import itertools
import sklearn.metrics as metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ## SUPPORT VECTOR MACHINE
from sklearn.svm import SVC
# instantiate the model
svm = SVC(kernel='linear', C=1.0, random_state=12)
#fit the model
svm.fit(x_train, y_train)

#predicting the target value from the model for the samples
y_test_svm = svm.predict(x_test)
y_train_svm = svm.predict(x_train)

# ### CONFUSION MATRIX
confusion_matrix_svm = confusion_matrix(y_test, y_test_svm)
print("Confusion Matrix - SVM")
print(confusion_matrix_svm)

# ### CLASSIFICATION REPORT
classification_report_svm = classification_report(y_test, y_test_svm)
print("Classification Report - SVM")
print(classification_report_svm)

# ### METRICS
tn, fp, fn, tp = confusion_matrix(y_test, y_test_svm).ravel()

accuracy  =(tp+tn)/(tp+tn+fp+fn)
precision =(tp)/(tp+fp)
recall  =(tp)/(tp+fn)
f1 =2*(( precision * recall)/( precision + recall))

print('Accuracy:\t',accuracy*100,
    '\nPrecision:\t',precision*100,
    '\nRecall: \t',recall*100,
    '\nF1-Score:\t',f1*100)

# ## KNN
from sklearn.neighbors import KNeighborsClassifier  
knn= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
knn.fit(x_train, y_train)

#predicting the target value from the model for the samples
y_test_knn = knn.predict(x_test)
y_train_knn = knn.predict(x_train)

# ### CONFUSION MATRIX
confusion_matrix_knn = confusion_matrix(y_test, y_test_knn)
print("Confusion Matrix - KNN")
print(confusion_matrix_knn)

# ### CLASSIFICATION REPORT
classification_report_knn = classification_report(y_test, y_test_knn)
print("Classification Report - KNN")
print(classification_report_knn)

# ### METRICS
tn, fp, fn, tp = confusion_matrix(y_test, y_test_knn).ravel()

accuracy  =(tp+tn)/(tp+tn+fp+fn)
precision =(tp)/(tp+fp)
recall =(tp)/(tp+fn)
f1 =2*(( precision_score * recall_score)/( precision_score + recall_score))

print('Accuracy:\t',accuracy*100,
    '\nPrecision:\t',precision*100,
    '\nRecall: \t',recall*100,
    '\nF1-Score:\t',f1*100)