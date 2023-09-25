from CalculateDistance import *

def calculate_mean(values):
    if len(values) == 0:
        return 0
    return sum(values) / len(values)



def fit_kmeans(X,k):
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    # print(k, centroids)
    #label = np.argmin(cdist(X,centroids), axis=1)
    
    labels =[]
    for i in X:
        labels.append(calculate_distance_from_centroids(i, centroids))
    # print(k,labels)
    
    
    cluster= [] * k
    for c in range(100):
         #p_labels=label.copy()
         p_label=labels

         for r in range(k):
            for index,i in enumerate(labels):
                if i==r:
                 cluster.append(X[index])
            centroids[r]=calculate_mean(cluster)
            #print(centroids[r])
            

         #centroid=np.array([np.mean(X[label==r],axis=0) for r in range(k)])

        # Calculate the sum of squared error (SSE) for the K-Means clustering
            sse = 0
            for cluster_idx, cluster_data_points in enumerate(cluster):
                for data_point_idx in cluster_data_points:
                    sse += np.sum((X[data_point_idx] - centroids[cluster_idx]) ** 2)
            
            print(sse)
            cluster.clear()

         labels = []
         for i in X:
            labels.append(calculate_distance_from_centroids(i, centroids))
        
         print(labels)

         #label = np.argmin(cdist(X, centroids), axis=1)

         if p_label==labels:
             break

    cluster = [[] for x in range(k)]
    for r in range(k):
        for index, i in enumerate(labels):
            if i == r:
                cluster[r].append(index)
    return cluster,centroids,labels