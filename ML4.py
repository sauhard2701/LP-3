# ASSIGNMENT-4
# Given a bank customer, build a neural network-based classifier

import numpy as np
import pandas as pd
#-----------------------------#
df = pd.read_csv('churn_Modelling.csv', index_col='RowNumber')
df.head()
#-----------------------------#
df.info()
#-----------------------------#
# Check for null values
df.isnull().values.any()
#-----------------------------#
df.describe()
#-----------------------------#
x_columns = df.columns.tolist()[2:12]
y_columns = df.columns.tolist()[-1:]
#-----------------------------#
print(f'All columns: {df.columns.tolist()}')
#-----------------------------#
print(f'X values: {x_columns}')
print(f'y values: {y_columns}')
#-----------------------------#
x = df[x_columns].values # Credit Score through Estimated Salary
y = df[y_columns].values # Exited
#-----------------------------#
# ## PREPROCESSING
# ### LABEL ENCODING
from sklearn.preprocessing import LabelEncoder
#-----------------------------#
print(x[:8,1], '... will now become: ')

label_x_country_encoder = LabelEncoder()
x[:,1] = label_x_country_encoder.fit_transform(x[:,1])
print(x[:8,1])
#-----------------------------#
print(x[:6,2], '... will now become: ')

label_x_gender_encoder = LabelEncoder()
x[:,2] = label_x_gender_encoder.fit_transform(x[:,2])
print(x[:6,2])
 #-----------------------------#
# ### FEATURE SCALING
# Feature scaling is a method used to standardize the range of independent variables or features of data. It is basically scaling all the dimensions to be even so that one independent variable does not dominate another. For example, bank account balance ranges from millions to 0, whereas gender is either 0 or 1. If one of the features has a broad range of values, the distance will be governed by this particular feature. Therefore, the range of all features should be normalized so that each feature contributes approximately proportionately to the final distance.

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
#-----------------------------#

pipeline = Pipeline(
    [('Categorizer', ColumnTransformer(
         [ # Gender
          ("Gender Label encoder", OneHotEncoder(categories='auto', drop='first'), [2]),
           # Geography
          ("Geography One Hot", OneHotEncoder(categories='auto', drop='first'), [1])
         ], remainder='passthrough', n_jobs=1)),
     # Standard Scaler for the classifier
    ('Normalizer', StandardScaler())
    ])
#-----------------------------#
x = pipeline.fit_transform(x)
#-----------------------------#
# ## Making the NN
from keras.models import Sequential
from keras.layers import Dense, Dropout
#-----------------------------#
# Initializing the ANN
classifier = Sequential()
#-----------------------------#
# Splitting the dataset into the Training and Testing set.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)
#-----------------------------#
print("The shape of X_train is:",x_train.shape)
print("The shape of X_test is:",x_test.shape)
print("The shape of y_train is:",y_train.shape)
print("The shape of y_test is:",y_test.shape)
#-----------------------------#
# ### ADDING INPUT LAYER
# This adds the input layer (by specifying input dimension) AND the first hidden layer (units)
classifier.add(Dense(6, activation = 'relu', input_shape = (x_train.shape[1], )))
classifier.add(Dropout(rate=0.1)) 
#-----------------------------#
# ### ADDING 2ND HIDDEN LAYER
# We will make our second hidden layer also have 6 nodes, just playing with the same arithmetic we used to determine the dimensions of the first hidden layer (average of your input and output layers) $(11+1)\div 2 = 6 $.
# Notice that we do not need to specify input dim. 
classifier.add(Dense(6, activation = 'relu')) 
classifier.add(Dropout(rate=0.1))
#-----------------------------#
### Adding the output layer
# Notice that we do not need to specify input dim. 
# we have an output of 1 node, which is the the desired dimensions of our output (stay with the bank or not)
# We use the sigmoid because we want probability outcomes
classifier.add(Dense(1, activation = 'sigmoid')) 
classifier.summary()
#-----------------------------#
# ## Compiling the Neural Network
classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
#-----------------------------#
# ## Fitting the Neural Network
history = classifier.fit(x_train, y_train, batch_size=32, epochs=200, validation_split=0.1, verbose=2)
#-----------------------------#
# ## TESTING THE NN
y_pred = classifier.predict(x_test)
print(y_pred[:5])
#-----------------------------#
y_pred = (y_pred > 0.5).astype(int)
print(y_pred[:5])
#-----------------------------#
# ## REPORTS
from sklearn.metrics import classification_report, confusion_matrix
#-----------------------------#
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(cm)
#-----------------------------#
cr = classification_report(y_test, y_pred)
print("Classification Report")
print(cr)
#-----------------------------#
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy  =(tp+tn)/(tp+tn+fp+fn)
precision =(tp)/(tp+fp)
recall  =(tp)/(tp+fn)
f1_score =2*(( precision * recall)/( precision + recall))
#-----------------------------#
print( 
    'Accuracy:\t',accuracy*100,
    '\nPrecision:\t',precision*100,
    '\nRecall: \t',recall*100,
    '\nF1-Score:\t',f1_score*100)