import pandas as pd
import numpy as np
#-----------------------------#
df = pd.read_csv('spam_ham_dataset.csv')
#-----------------------------#
df.head()
#-----------------------------#
df.dtypes
#-----------------------------#
# Check for null values
df.isnull().values.any()
#-----------------------------#
df.describe()
#-----------------------------#
x = df['text']
x
#-----------------------------#
y = df['label']
y
#-----------------------------#
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 1)
#-----------------------------#
print("The shape of X_train is:",x_train.shape)
print("The shape of X_test is:",x_test.shape)
print("The shape of y_train is:",y_train.shape)
print("The shape of y_test is:",y_test.shape)
#-----------------------------#
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
#-----------------------------#
X_train_tfidf = vectorizer.fit_transform(x_train)
X_train_tfidf.shape
#-----------------------------#
from sklearn.svm import LinearSVC
clf_svm = LinearSVC()
clf_svm.fit(X_train_tfidf,y_train)
#-----------------------------#
from sklearn.pipeline import Pipeline
text_clf_svm = Pipeline([('tfidf', TfidfVectorizer()),
                        ('clf_svm', LinearSVC()),])
text_clf_svm.fit(x_train, y_train)  
y_test_svm = text_clf_svm.predict(x_test)
#-----------------------------#
# ## KNN
from sklearn.neighbors import KNeighborsClassifier 
clf_knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
clf_knn.fit(X_train_tfidf,y_train)
#-----------------------------#
from sklearn.pipeline import Pipeline
text_clf_knn = Pipeline([('tfidf', TfidfVectorizer()),
                        ('clf_knn', KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)),])
text_clf_knn.fit(x_train, y_train)  
y_test_knn = text_clf_knn.predict(x_test)
#-----------------------------#
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
#-----------------------------#
# ### CONFUSION MATRIX SVM
confusion_matrix_svm = confusion_matrix(y_test, y_test_svm)
print("Confusion Matrix - SVM")
print(confusion_matrix_svm)
#-----------------------------#
# ### CONFUSION MATRIX KNN
confusion_matrix_knn = confusion_matrix(y_test, y_test_knn)
print("Confusion Matrix - KNN")
print(confusion_matrix_knn)
#-----------------------------#
# ### CLASSIFICATION REPORT SVM
classification_report_svm = classification_report(y_test, y_test_svm)
print("Classification Report - SVM")
print(classification_report_svm)
#-----------------------------#
# ### CLASSIFICATION REPORT KNN
classification_report_knn = classification_report(y_test, y_test_knn)
print("Classification Report - KNN")
print(classification_report_knn)
#-----------------------------#
# ### METRICS SVM
tn, fp, fn, tp = confusion_matrix(y_test, y_test_svm).ravel()
accuracy  =(tp+tn)/(tp+tn+fp+fn)
precision =(tp)/(tp+fp)
recall  =(tp)/(tp+fn)
f1 =2*(( precision * recall)/( precision + recall))
print('SVM METRICS',
    '\nAccuracy:\t',accuracy*100,
    '\nPrecision:\t',precision*100,
    '\nRecall: \t',recall*100,
    '\nF1-Score:\t',f1*100)
#-----------------------------#
# ### METRICS KNN
tn, fp, fn, tp = confusion_matrix(y_test, y_test_knn).ravel()
accuracy  =(tp+tn)/(tp+tn+fp+fn)
precision =(tp)/(tp+fp)
recall =(tp)/(tp+fn)
f1 =2*(( precision * recall)/( precision + recall))
print('KNN METRICS',
    '\nAccuracy:\t',accuracy*100,
    '\nPrecision:\t',precision*100,
    '\nRecall: \t',recall*100,
    '\nF1-Score:\t',f1*100)