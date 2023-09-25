import numpy as np


# Define a function to calculate Euclidean distance between two points
def calculate_euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

# Define a function to calculate the distance between a data point and a set of centroids
def calculate_distance_from_centroids(data_point, centroids):
    distances_to_centroids = [calculate_euclidean_distance(data_point, centroid) for centroid in centroids]
    closest_centroid_index = np.argmin(distances_to_centroids)
    return closest_centroid_index



def calculate_intra_cluster_distance(cluster_points, X):
    """
    Calculate the minimum distance between all pairs of data points within a cluster.
    """
    if len(cluster_points) < 2:
        return np.inf
    
    # Initialize the minimum distance with a large value
    min_distance = np.inf

    # Iterate over all pairs of data points within the cluster
    for i in cluster_points:
        for j in cluster_points:
            if i != j:
                # Calculate the Euclidean distance between data points i and j
                distance = calculate_euclidean_distance(X[i], X[j])
                
                # Update the minimum distance if a smaller distance is found
                if distance < min_distance and distance != 0:
                    min_distance = distance
    
    return min_distance


def calculate_inter_cluster_distance(centroid1, centroid2):
    """
    Calculate the distance between the centroids of two clusters.
    """
    return np.linalg.norm(centroid1 - centroid2)