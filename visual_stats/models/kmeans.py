import numpy as np

from scipy.spatial.distance import euclidean
from typing import Tuple, Union


class KMeans():
    def __init__(self, points: np.ndarray, clusters: int):
        self.points = points
        self.clusters = clusters

        self.centroids = [points[idx] \
                          for idx in np.random.choice(len(points), clusters,
                                                      replace=False)]

    def predict(self, x: Union[Tuple[int], np.ndarray]) -> int:
        distances = [euclidean(x, centroid) for centroid in self.centroids]

        return np.argmin(distances)

    def train(self, x: np.ndarray):
        clusters = np.asarray([self.predict(point) for point in x])

        self.centroids = [np.mean(x[clusters == i], axis=0) \
                          for i in range(self.clusters)]
