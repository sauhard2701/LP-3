# ASSIGNMENT-1
# Linear Regression
#-----------------------------#
import pandas as pd
import numpy as np
#-----------------------------#
df= pd.read_csv('uber.csv')
df.head()
#-----------------------------#
df.dtypes
#-----------------------------#
# Check for null values
df.isnull().values.any()
#-----------------------------#
df.dropna(inplace=True)
#-----------------------------#
df.drop(['Unnamed: 0','key','pickup_datetime'], axis=1, inplace=True)
#-----------------------------#
df.info()
#-----------------------------#
# Removing all the rows with values = 0
df = df[(df[['fare_amount','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count']] != 0).all(axis=1)]
#-----------------------------#
df.info()   
#-----------------------------#
from math import *
#-----------------------------#
def distance_transform(longitude1, latitude1, longitude2, latitude2):
    travel_dist = []
    
    for pos in range(len(longitude1)):
        long1,lati1,long2,lati2 = map(radians,[longitude1[pos],latitude1[pos],longitude2[pos],latitude2[pos]])
        dist_long = long2 - long1
        dist_lati = lati2 - lati1
        a = sin(dist_lati/2)**2 + cos(lati1) * cos(lati2) * sin(dist_long/2)**2
        c = 2 * asin(sqrt(a))*6371
        travel_dist.append(c)
       
    return travel_dist
#-----------------------------#
df['dist_travel_km'] = distance_transform(df['pickup_longitude'].to_numpy(),
                                                df['pickup_latitude'].to_numpy(),
                                                df['dropoff_longitude'].to_numpy(),
                                                df['dropoff_latitude'].to_numpy()
                                              )

#-----------------------------#
df.head()
#-----------------------------#
df.describe().transpose()
#-----------------------------#
df.columns[df.dtypes == 'object']
#-----------------------------#
df.fare_amount.min()
#-----------------------------#
import matplotlib.pyplot as plt
import seaborn as sns
#-----------------------------#
plt.figure(figsize=(20,30))
for i , variable in enumerate(df.iloc[:, 0::]):
    plt.subplot(6,5,i+1)
    plt.boxplot(df[variable])
    plt.tight_layout()
    plt.title(variable)
plt.show()
#-----------------------------#
# ### Removing Outliers
def remove_outlier(df1 , col):
    Q1 = df1[col].quantile(0.25)
    Q3 = df1[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_whisker = Q1-1.5*IQR
    upper_whisker = Q3+1.5*IQR
    df1[col] = np.clip(df1[col] , lower_whisker , upper_whisker)
    return df1

def treat_outliers_all(df1 , col_list):
    for c in col_list:
        df1 = remove_outlier(df1 , c)
    return df1
#-----------------------------#
df = treat_outliers_all(df , df.iloc[:, 0::])
#-----------------------------#
plt.figure(figsize=(20,30))
for i , variable in enumerate(df.iloc[:, 0::]):
    plt.subplot(6,5,i+1)
    plt.boxplot(df[variable])
    plt.tight_layout()
    plt.title(variable)
plt.show()
#-----------------------------#
# ### Choosing x and y
y = df.iloc[:, 0:1]
y.info()
#-----------------------------#
y.head()
#-----------------------------#
x = df.drop('fare_amount',axis = 1)
x.info()
#-----------------------------#
x.head()
#-----------------------------#
# ### Train-Test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

print("The shape of X_train is:",x_train.shape)
print("The shape of X_test is:",x_test.shape)
print("The shape of y_train is:",y_train.shape)
print("The shape of y_test is:",y_test.shape)
#-----------------------------#
# ### Linear Regression  Model
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse
#-----------------------------#
linreg_full = sm.OLS(y_train, x_train).fit()
linreg_full_predictions = linreg_full.predict(x_test)
linreg_full_predictions.head()
#-----------------------------#
actual_fare = y_test["fare_amount"]
actual_fare.head()
#-----------------------------#
# calculate rmse using rmse()
linreg_full_rmse = rmse(actual_fare,linreg_full_predictions )
#-----------------------------#
# calculate R-squared using rsquared
linreg_full_rsquared = linreg_full.rsquared
#-----------------------------#
# calculate Adjusted R-Squared using rsquared_adj
linreg_full_rsquared_adj = linreg_full.rsquared_adj 
#-----------------------------#
print('Model: Linreg full model',
      '\nRMSE:  ',linreg_full_rmse,
      '\nR-Squared: ', linreg_full_rsquared,
      '\nAdj. R-Squared:  ', linreg_full_rsquared_adj)
#-----------------------------#
a= linreg_full.predict([-73.999817,40.738354,-73.999512,40.723217,1.0,1.683323])
print(a)