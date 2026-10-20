# loads dataset
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import numpy as np

iris = datasets.load_iris()

X = iris['data']
y = iris['target']

# performs clustering with k = 3
model = KMeans(n_clusters=3, random_state=42, n_init=10)

model.fit(X)

clusters = model.labels_
centroids = model.cluster_centers_

print("\nCluster Centroids:")
print(centroids)

# computes RSME for each cluster
for i in range(3):

    cluster_points = X[clusters == i]

    centroid = centroids[i]

    rmse = np.sqrt(
        np.mean(np.sum((cluster_points - centroid) ** 2, axis=1))
    )

    print(f"\nCluster {i}:")
    print(f"Number of points: {len(cluster_points)}")
    print(f"RMSE: {rmse:.4f}")

# comparison between clusters and results from mlp_classifier.py
