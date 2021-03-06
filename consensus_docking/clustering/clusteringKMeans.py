import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class ClusteringKMeans:
    def __init__(self, df, n_clusters=8, initial_clusters='k-means++', max_iter=300, n_init=10, data_weight=None):
        """
        It initializes a ClusteringKMeans object.
        Parameters
        ----------
        df : Pandas DataFrame
            DataFrame containing the data to be used for the clustering.
            Shape (n_samples, n_features)
        n_clusters : int
            The number of clusters to form as well as the number of centroids to generate.
        initial_clusters : str or array (n_cluster, n_features)
            ‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
            ‘random’: choose n_clusters observations (rows) at random from data for the initial centroids.
            If an array is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.
        max_iter : int
            Maximum number of iterations of the k-means algorithm for a single run.
        n_init : int
            Number of time the k-means algorithm will be run with different centroid seeds. The final results will be
            the best output of n_init consecutive runs in terms of inertia.
        data_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in df. If None, all observations are assigned equal weight.
        """
        self.data = df
        self.data_weight = data_weight
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.labels = None
        self.centroids = None
        self.initial_clusters = initial_clusters

        self.model = KMeans(n_clusters=self.n_clusters, init=self.initial_clusters, max_iter=max_iter, n_init=n_init)

    def fit(self):
        """
        Fits the model with self.data and, in case weights are specify it uses them for the clustering. The fitted model
        is returned and saved in self.model.
        Returns
        -------
        KMeans sklearn object model fitted.
        """
        if self.data_weight is None:
            self.model.fit(self.data.values)
        else:
            if type(self.data_weight) is np.ndarray:
                self.model.fit(self.data.values, sample_weight=self.data_weight)
            elif type(self.data_weight) is pd.core.frame.DataFrame:
                self.model.fit(self.data.values, sample_weight=self.data_weight.values)
        if self.model.n_iter_ < self.max_iter:
            print("KMeans converged!")
        else:
            print("Warning: KMeans did not converge!!")

        return self.model

    def get_labels(self):
        """
        Get cluster labels after fitting the model.
        Returns
        -------
        List with the labels of the cluster. Cluster -1 contains the outliers.
        """
        self.labels = self.model.labels_
        return self.labels

    def get_centroids(self):
        """
        Finds the centroid indexes in self.data.values with the centroid coordinates given by the fitted model.
        Returns
        -------
        Numpy array with centroids indexes.
        """
        from sklearn.metrics import pairwise_distances_argmin_min
        centroids_coord = self.model.cluster_centers_
        self.centroids, _ = pairwise_distances_argmin_min(centroids_coord, self.data.values)
        return self.centroids

    def print_info(self):
        """
        Reports the parameters used for the clustering.
        """
        print("\n#################")
        print("Running KMeans Algorithm with parameters:")
        print(f"  - n_custers \t\t {self.n_clusters}")
        print(f"  - n_init \t\t {self.n_init}")
        print(f"  - max_iter \t\t {self.max_iter}")
        print(f"  - initial_cluster \t {self.initial_clusters}")
        if self.data_weight is None:
            print("  - Weights for each pose were not specified, so they all have the same weight.")
        else:
            print("  - Using weights for each pose.")
        print("#################\n")
