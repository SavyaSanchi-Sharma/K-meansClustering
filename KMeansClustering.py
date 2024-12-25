import numpy as np
import matplotlib.pyplot as plt

def mahalanobis_dist(x1, x2, cov_matrix):
    mean_diff = x1 - x2  
    cov_matrix_inv = np.linalg.inv(cov_matrix)  
    distance = np.sqrt(np.dot(np.dot(mean_diff.T, cov_matrix_inv), mean_diff))
    return distance

   
def KMeansClustering(X,K,maxIteration,mod:bool,tolerance=1e-4):
    m=X.shape[0]
    np.random.seed(42)
    centroids=X[np.random.choice(m,K,replace=False)]
    idx=np.zeros(m,dtype=int)
    prevCentroid=np.zeros_like(centroids)
    cov=np.cov(X,rowvar=False)
    print("Covariance Matrix:")
    print(cov)
    print("\nMean of the data:")
    print(np.mean(X, axis=0))
    iteration_list = []
    centroid_movements = []
    for i in range(maxIteration):
        if mod:
            dist = np.zeros((m, K))
            for j in range(m):
                for k in range(K):
                    dist[j, k] = mahalanobis_dist(X[j], centroids[k], cov)
        else: 
            dist=np.linalg.norm(X[:,np.newaxis]-centroids,axis=2)
        idx=np.argmin(dist,axis=1)
        prevCentroid=centroids.copy()
        for k in range(K):
            points=X[idx==k]
            if len(points)>0:
                centroids[k]=np.mean(points,axis=0)
        centroid_movement = np.linalg.norm(centroids - prevCentroid)
        iteration_list.append(i + 1)
        centroid_movements.append(centroid_movement)
        if centroid_movement < tolerance:
            print(f"Converged in {i + 1} iterations!")
            break
    plt.figure(figsize=(8, 5))
    plt.plot(iteration_list, centroid_movements, marker='o', label="Centroid Movement")
    plt.title("Centroid Movement vs. Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Centroid Movement")
    plt.grid(True)
    plt.legend()
    plt.show()
        
    return centroids, idx


