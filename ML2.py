# ASSIGNMENT-2
# Implement K-Means clustering/ hierarchical clustering on sales_data_sample.csv dataset.

#importing the required libraries
import pandas as pd
import numpy as np
#viz Libraries
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
#datetime
import datetime as dt
#StandardSccaler
from sklearn.preprocessing import StandardScaler
#KMeans
from sklearn.cluster import KMeans
#-----------------------------#
df = pd.read_csv('sales_data_sample.csv', encoding = 'unicode_escape')
#-----------------------------#
df.shape #Dimensions of the data
#-----------------------------#
df.head() #Glimpse of the data
#-----------------------------#
#Removing the variables which dont add significant value fot the analysis.
to_drop = ['PHONE','ADDRESSLINE1','ADDRESSLINE2','STATE','POSTALCODE']
df = df.drop(to_drop, axis=1)
#-----------------------------#
df.isnull().sum()
#-----------------------------#
df.dtypes
#-----------------------------#
df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
#-----------------------------#
df['ORDERDATE'] = [d.date() for d in df['ORDERDATE']]
df.head()
#-----------------------------#
# Calculate Recency, Frequency and Monetary value for each customer
latest_date = df['ORDERDATE'].max() + dt.timedelta(days=1) #latest date in the data set
df_RFM = df.groupby(['CUSTOMERNAME'])

df_RFM = df_RFM.agg({
    'ORDERDATE': lambda x: (latest_date - x.max()).days,
    'ORDERNUMBER': 'count',
    'SALES':'sum'})

#Renaming the columns
df_RFM.rename(columns={'ORDERDATE': 'Recency',
                   'ORDERNUMBER': 'Frequency',
                   'SALES': 'MonetaryValue'}, inplace=True)
#-----------------------------#
data = df_RFM[['Recency','Frequency','MonetaryValue']]
data.head()
#-----------------------------#
plt.figure(figsize=(10,6))

plt.subplot(1,3,1)
sns.histplot(data['Recency'], kde=True)

plt.subplot(1,3,2)
sns.histplot(data['Frequency'], kde=True)

plt.subplot(1,3,3)
plt.xticks(rotation = 45)
sns.histplot(data['MonetaryValue'], kde=True)

plt.title('Distribution of Recency, Frequency and MonetaryValue')
plt.legend()
plt.show()
#-----------------------------#
data_log = np.log(data)
#-----------------------------#
data_log.head()
#-----------------------------#
plt.figure(figsize=(10,6))

plt.subplot(1,3,1)
sns.histplot(data_log['Recency'], kde=True)

plt.subplot(1,3,2)
sns.histplot(data_log['Frequency'], kde=True)

plt.subplot(1,3,3)
sns.histplot(data_log['MonetaryValue'], kde=True)

plt.title('Distribution of Recency, Frequency and MonetaryValue after Log Transformation')
plt.legend()
plt.show()
#-----------------------------#
# Initialize a scaler
scaler = StandardScaler()
#-----------------------------#
# Fit the scaler
scaler.fit(data_log)
#-----------------------------#
# Scale and center the data
data_normalized = scaler.transform(data_log)
#-----------------------------#
# Create a pandas DataFrame
data_normalized = pd.DataFrame(data_normalized, index=data_log.index, columns=data_log.columns)
#-----------------------------#
# Print summary statistics
data_normalized.describe().round(2)
#-----------------------------#
# # Choosing number of Clusters using Elbow Method
# Fit KMeans and calculate SSE for each k
sse={}
for k in range(1, 21):
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(data_normalized)
    sse[k] = kmeans.inertia_ 
#-----------------------------#
plt.figure(figsize=(10,6))
plt.title('The Elbow Method')

# Add X-axis label "k"
plt.xlabel('k')

# Add Y-axis label "SSE"
plt.ylabel('SSE')

# Plot SSE values for each key in the dictionary
sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
plt.text(4.5,60,"Largest Angle",bbox=dict(facecolor='lightgreen', alpha=0.5))
plt.show()
#-----------------------------#
# # Running KMeans with 5 clusters
# Initialize KMeans
kmeans = KMeans(n_clusters=5, random_state=1) 
#-----------------------------#
# Fit k-means clustering on the normalized data set
kmeans.fit(data_normalized)
#-----------------------------#
# Extract cluster labels
cluster_labels = kmeans.labels_
#-----------------------------#
# Assigning Cluster Labels to Raw Data
# Create a DataFrame by adding a new cluster label column
data_rfm = data.assign(Cluster=cluster_labels)
data_rfm.head()
#-----------------------------#
# Group the data by cluster
grouped = data_rfm.groupby(['Cluster'])
#-----------------------------#
# Calculate average RFM values and segment sizes per cluster value
grouped.agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'MonetaryValue': ['mean', 'count']
  }).round(1)

#----------------EXTRA--------------#
# # Calculating relative importance of each attribute
# Calculate average RFM values for each cluster
cluster_avg = data_rfm.groupby(['Cluster']).mean() 
print(cluster_avg)
#-----------------------------#
# Calculate average RFM values for the total customer population
population_avg = data.mean()
print(population_avg)
#-----------------------------#
# Calculate relative importance of cluster's attribute value compared to population
relative_imp = cluster_avg / population_avg - 1
#-----------------------------#
# Print relative importance score rounded to 2 decimals
print(relative_imp.round(2))
#-----------------------------#
#Plot Relative Importance
# Initialize a plot with a figure size of 8 by 2 inches 
plt.figure(figsize=(8, 2))

# Add the plot title
plt.title('Relative importance of attributes')

# Plot the heatmap
sns.heatmap(data=relative_imp, annot=True, fmt='.2f', cmap='RdYlGn')
plt.show()
