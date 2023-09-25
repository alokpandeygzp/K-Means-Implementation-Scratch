from itertools import combinations
from CalculateDistance import *


def calculate_silhouette_score(data, cluster_assignments):
    """
    Calculate the Silhouette score to assess clustering quality.
    """
    num_samples, num_features = data.shape
    unique_clusters = np.unique(cluster_assignments)
    num_clusters = unique_clusters.shape[0]
    
    
    silhouette_scores = np.zeros(num_samples)

    for i in range(num_samples):


        # Calculate the mean distance between the i-th data point and all other points in its cluster
        current_cluster = cluster_assignments[i]
        data_point = data[i]

        # Initialize an empty list to store distances
        distances_to_data_point = []

        # Loop through all data points and calculate distances
        for j in range(num_samples):
            if cluster_assignments[j] == current_cluster:
                distance_to_data_point = calculate_euclidean_distance(data_point, data[j])
                distances_to_data_point.append(distance_to_data_point)

        # Calculate the mean of the distances
        a_i = np.mean(distances_to_data_point)



        # Calculate the mean distance between the i-th data point and all other points in the nearest neighboring cluster
        b_i = np.inf
        for k in range(num_clusters):
            if k != current_cluster:

                # Initialize an empty list to store distances for data points in cluster k
                distances_to_data_points_in_cluster_k = []

                # Iterate through all data points
                for j in range(num_samples):

                    # Check if the data point belongs to cluster k
                    if cluster_assignments[j] == k:

                        # Calculate the Euclidean distance between the current data point and data_point
                        distance_to_data_point_k = calculate_euclidean_distance(data_point, data[j])
                        
                        # Append the distance to the list
                        distances_to_data_points_in_cluster_k.append(distance_to_data_point_k)

                # Calculate the mean of distances in cluster k
                b_ik = np.mean(distances_to_data_points_in_cluster_k)

                if b_ik < b_i:
                    b_i = b_ik
        
        # Calculate the silhouette coefficient for the i-th data point
        silhouette_scores[i] = (b_i - a_i) / max(b_i, a_i)
    
    # Calculate the average silhouette coefficient for all data points
    average_silhouette_score = np.mean(silhouette_scores)
    return average_silhouette_score



def dunn_index(X, clusters, centroids):
    """
    Calculate the Dunn Index for a set of clusters.
    A metric for evaluating clustering algorithms.
    """
    num_clusters = len(clusters)
    
    # Check if there are at least 2 clusters
    if num_clusters < 2:
        return 0.0  # Dunn Index is undefined with fewer than 2 clusters

    # Calculate the intra-cluster distances for each cluster
    intra_cluster_distances = [calculate_intra_cluster_distance(cluster, X) for cluster in clusters]

    # Find the maximum intra-cluster distance
    max_intra_cluster_distance = max(intra_cluster_distances)

    # Calculate the inter-cluster distances between all pairs of centroids
    inter_cluster_distances = [calculate_inter_cluster_distance(centroids[i], centroids[j])
                               for i, j in combinations(range(num_clusters), 2)]

    # Find the minimum inter-cluster distance
    min_inter_cluster_distance = min(inter_cluster_distances)

    # Calculate the Dunn Index value
    dunn_index_value = min_inter_cluster_distance / max_intra_cluster_distance

    return dunn_index_value


