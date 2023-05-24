import pandas as pd
import numpy as np

# load the crime data from the csv file
crime_data = pd.read_csv('crime_data.csv', index_col=0)

# ask the user to input the number of clusters (k)
number_of_clusters= int(input('Enter the number of clusters: '))

# initialize the centroids randomly
centroids = crime_data.sample(number_of_clusters)

# define a function to calculate the Manhattan distance between two points
def manhattan_distance(point1, point2):
    return np.abs(point1 - point2).sum()

# define a function to assign each data point to the closest centroid
def assign_clusters(data, centroids):
    clusters = []
    for i in range(len(data)):
        distances = [manhattan_distance(data.iloc[i], centroids.iloc[j]) for j in range(len(centroids))]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return clusters

# define a function to update the centroids based on the mean of the assigned data points
def update_centroids(data, clusters, centroids):
    for i in range(len(centroids)):
        cluster_data = data.iloc[[j for j in range(len(data)) if clusters[j] == i]]
        centroids.iloc[i] = cluster_data.mean()


def detect_outliers(data, clusters):
    outliers = []
    for i in range(len(clusters)):
        cluster_data = data.iloc[[j for j in range(len(data)) if clusters[j] == i]]
        q1 = cluster_data.quantile(0.25)
        q3 = cluster_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers.extend(cluster_data[(cluster_data < lower_bound) | (cluster_data > upper_bound)].dropna(how='all').index.tolist())
    return outliers


def printClusters():
    Clusters=[]
    for i in range(number_of_clusters):
       cluster_data = crime_data.iloc[[j for j in range(len(crime_data)) if clusters[j] == i]]
       print('\nCluster', i+1)
       print(cluster_data.index.tolist(),'\n')
       Clusters.append(cluster_data)
    return Clusters


# run the k-means algorithm for a fixed number of iterations
max_iterations = 100
for i in range(max_iterations):
    # assign data points to clusters
    clusters = assign_clusters(crime_data, centroids)
    
    # update centroids
    update_centroids(crime_data, clusters, centroids)
    
    # detect outliers
    outliers = detect_outliers(crime_data, clusters)
    
    # check for convergence
    if i > 0 and all(x == y for x, y in zip(clusters, prev_clusters)):
        print('Ended after', i+1, 'iterations')
        break
        
    prev_clusters = clusters.copy()


clusters=printClusters() 
if len(outliers) > 0:
    print('\n Outliers:', outliers,'\n')


     









