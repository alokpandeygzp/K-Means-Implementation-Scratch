import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from ClusterValidityIndices import *
from KMeans import *



df = pd.read_csv("IRIS.csv")
# X = df.drop(['species'], axis=1)
X = df.iloc[:, :-1]
X = (X - X.mean())/X.std()
Y = X
X = X.values


""" *********************************** (I) Elbow Method ************************************** """

# Initialize a list to store the cost values
cost_values = []

# Running a loop for the elbow method with a range of cluster counts
for num_clusters in range(2, 6):

    clusters, cluster_centroids, cluster_labels = fit_kmeans(X, num_clusters)

    # Calculate the total cost (sum of squared distances) for each cluster
    total_cost = 0
    for cluster_idx, cluster_data_points in enumerate(clusters):
        for data_point_idx in cluster_data_points:
            total_cost += np.sum((X[data_point_idx] - cluster_centroids[cluster_idx]) ** 2)
    
    cost_values.append(total_cost)


# Plot the cost values against the number of clusters
plt.plot(range(2, 6), cost_values, marker='o')
plt.title('Elbow Method for Optimal Cluster Count')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.show()





""" *********************************** (II) Dunn Index ************************************** """

# Initialize a list to store the Dunn index values
dunn_values = []

# Running a loop for different cluster counts
for num_clusters in range(2, 6):

    clusters, cluster_centroids, cluster_labels = fit_kmeans(X, num_clusters)

    # Calculate the Dunn index for the current clustering
    dunn_index_value = dunn_index(X, clusters, cluster_centroids)
    
    dunn_values.append(dunn_index_value)

# Plotting the Dunn index graph
plt.plot(range(2, 6), dunn_values, marker='o')
plt.title('Dunn Index Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Dunn Index Value')
plt.show()





""" *********************************** (III) Silhouette Score ************************************** """


# Define the number of runs and the range of cluster values
num_runs = 10

# Lists to store validity indices
average_silhouette_scores = []

# Loop over different cluster values
for num_clusters in range(2, 6):
    
    # Initialize variables to accumulate scores
    total_silhouette_score = 0
    
    # Perform multiple runs
    for _ in range(num_runs):
        
        clusters, cluster_centroids, cluster_labels = fit_kmeans(X, num_clusters)
        total_silhouette_score += calculate_silhouette_score(X, cluster_labels)
    
    # Calculate average scores for this cluster value
    avg_silhouette_score = total_silhouette_score / num_runs
    
    # Append the average scores to the respective lists
    average_silhouette_scores.append(avg_silhouette_score)


# Plot the Silhouette Score
plt.figure(figsize=(6, 4))
plt.plot(range(2, 6), average_silhouette_scores, marker='o')
plt.title('Silhouette Score vs. Number of Clusters (K)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.show()





""" *********************************** After performing PCA ************************************** """

# Define a function to plot the clusters and centroids
def plot_clusters(X, centroids, labels):
    
    # Get unique labels (cluster IDs)
    unique_labels = np.unique(labels)
    
    # Plot data points for each cluster
    for label in unique_labels:
        # Select data points belonging to the current cluster
        cluster_data = X[labels == label]
        
        # Plot the data points with the same label (cluster) using a unique color
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {label}')
    
    # Plot centroids with a star marker and a different color
    for label in unique_labels:
        plt.scatter(centroids[label, 0], centroids[label, 1], marker='*', s=100, color='k', label="Centroid")
    
    # Add a legend to distinguish clusters and centroids
    plt.legend()
    
    # Display the plot
    plt.show()


# Perform Principal Component Analysis (PCA) to reduce the data to two dimensions
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(X)

# Fit a K-Means clustering model to the reduced data
clusters, cluster_centroids, cluster_labels = fit_kmeans(reduced_data, 3)

# Calculate the sum of squared error (SSE) for the K-Means clustering
sse = 0
for cluster_idx, cluster_data_points in enumerate(clusters):
    for data_point_idx in cluster_data_points:
        sse += np.sum((reduced_data[data_point_idx] - cluster_centroids[cluster_idx]) ** 2)

# Print the SSE using our K-Means implementation
print("Sum of squared error using our K-Means (After performing PCA):", sse)

# Plot the clusters and centroids
plot_clusters(reduced_data, cluster_centroids, cluster_labels)
