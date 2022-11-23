# ASSIGNMENT-3
# Implement K-Nearest Neighbors algorithm on diabetes.csv dataset.

import pandas as pd
import numpy as np
#-----------------------------#
df= pd.read_csv('diabetes.csv')
#-----------------------------#
df.shape #Dimensions of the data
#-----------------------------#
df.head() #Glimpse of the data
#-----------------------------#
# Check for null values
df.isnull().values.any()
#-----------------------------#
df.describe()
#-----------------------------#
df.dtypes
#-----------------------------#
x = df.drop('Outcome', axis=1)
x.shape
x
#-----------------------------#
y = df['Outcome']
y.shape
y
#-----------------------------#
from sklearn.model_selection import train_test_split
#-----------------------------#
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 1)

#-----------------------------#
print("The shape of X_train is:",x_train.shape)
print("The shape of X_test is:",x_test.shape)
print("The shape of y_train is:",y_train.shape)
print("The shape of y_test is:",y_test.shape)
#-----------------------------#
#importing packages


from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier 
#-----------------------------#
knn= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
knn.fit(x_train, y_train)
#-----------------------------#
#predicting the target value from the model for the samples
y_test_knn = knn.predict(x_test)
y_train_knn = knn.predict(x_train)
#-----------------------------#
confusion_matrix_knn = confusion_matrix(y_test, y_test_knn)
print("Confusion Matrix - KNN")
print(confusion_matrix_knn)
#-----------------------------#
classification_report_knn = classification_report(y_test, y_test_knn)
print("Classification Report - KNN")
print(classification_report_knn)
#-----------------------------#
tn, fp, fn, tp = confusion_matrix(y_test, y_test_knn).ravel()
#-----------------------------#
accuracy  =(tp+tn)/(tp+tn+fp+fn)
precision =(tp)/(tp+fp)
recall  =(tp)/(tp+fn)
f1 =2*(( precision* recall)/( precision + recall))
#-----------------------------#
print('Accuracy:\t',accuracy*100,
    '\nPrecision:\t',precision*100,
    '\nRecall: \t',recall*100,
    '\nF1-Score:\t',f1*100)
