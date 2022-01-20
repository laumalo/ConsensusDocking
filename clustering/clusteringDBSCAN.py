import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


class ClusteringDBSCAN:
    def __init__(self, df, eps=3 * 8, metric='euclidean', metric_param=None, min_samples=5, data_weight=None,
                 n_jobs=None):
        """
        It initializes a ClusteringDBSCAN object.
        Parameters
        ----------
        df : Pandas DataFrame
            DataFrame containing the data to be used for the clustering.
            Shape (n_samples, n_features)
        eps : float
            The maximum distance between two samples for them to be considered as in the same
            neighborhood.
        metric : str
            The metric to use when calculating distance between instances in a feature array.
            It can be: [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’] and [‘braycurtis’, ‘canberra’, 
            ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘minkowski’, 
            ‘rogerstanimoto’, ‘russellrao’,‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]
        metric_param : array_like for mahalanobis/ int for minkowski / 1D array_like for seuclidean / otherwise None
            Extra parameter needed for some metrics.
            Mahalanobis:
                The inverse of the covariance matrix.
            Minkowski:
                When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2.
            Seuclidean:
                 1-D array of component variances. It is usually computed among a larger collection vectors.
        min_samples : int
            The number of samples (or total weight) in a neighborhood for a point to be
            considered as a core point. This includes the point itself.
        data_weight : array-like of shape (n_samples,), default=None
            Weight of each sample, such that a sample with a weight of at least min_samples is
            by itself a core sample; a sample with a negative weight may inhibit its
            eps-neighbor from being core. Note that weights are absolute, and default to 1.
        n_jobs: int
            The number of parallel jobs to run. None means 1. -1 means using all processors.
        """
        self.data = df
        self.eps = eps  # epsilon parameter in DBSCAN algorithm
        self.metric = metric.lower()
        self.metric_param = metric_param
        self.min_samples = min_samples
        self.data_weight = data_weight
        self.n_jobs = n_jobs
        self.labels = None
        self.centroids = None

        self.__validate_input()

        if metric == 'minkowski':
            assert type(metric_param) is int, "metric_param should be an int value: p = 1, this is equivalent to using" \
                                              " manhattan_distance (l1), and euclidean_distance (l2) for p = 2."

            self.model = DBSCAN(min_samples=self.min_samples, eps=self.eps, metric=self.metric, p=metric_param,
                                n_jobs=n_jobs)
        elif metric == 'mahalanobis':
            raise ModuleNotFoundError("Still not implemented. Try another metric.")
        elif metric == 'seuclidean':
            raise ModuleNotFoundError("Still not implemented. Try another metric.")
        else:
            self.model = DBSCAN(min_samples=self.min_samples, eps=self.eps, metric=self.metric, n_jobs=n_jobs)

    def __validate_input(self):
        """
        Verifies if the inputs are coherent, otherwise it will raise an error or print a warning.
        """
        valid_metrics = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan', 'braycurtis', 'canberra',
                         'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
                         'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
                         'sqeuclidean', 'yule']

        assert self.metric in valid_metrics, f"{self.metric} is not available, try with any of the following:" \
                                             f" {valid_metrics}."

        if self.data_weight is not None:
            assert self.data.shape[0] == self.data_weight.shape[
                0], f"Data weight (Shape: {self.data_weight.shape}) must have the same number of rows as the encoding" \
                    f" data (Shape: {self.data.shape})"

            if max(self.data_weight) >= self.min_samples:
                print(
                    "WARNING: There are poses that weight equal or more than min_samples, so a single pose can create"
                    " a cluster!")
        assert type(self.eps) == int, f"eps parameter must be and int value but found: {type(self.eps)}. " \
                                      f"Eps sets the maximum distance between two samples for them to be considered " \
                                      f"as in the same neighborhood."

    def fit(self):
        """
        Fits the model with self.data and, in case weights are specify it uses them for the clustering. The fitted model
        is returned and saved in self.model.
        Returns
        -------
        DBSCAN sklearn object model fitted.
        """
        if self.data_weight is None:
            self.model.fit(self.data.values)
        else:
            self.model.fit(self.data.values, sample_weight=self.data_weight.values)
        return self.model

    def get_labels(self):
        """
        Get cluster labels after fitting the model.
        Returns
        -------
        Numpy array with the labels of the cluster. Cluster -1 contains the outliers.
        """
        self.labels = self.model.labels_
        return self.labels

    def get_centroids(self):
        """
        It prints that DBSCAN cannot find centroids. Method added to be coherent with clusteringKMeans.
        Returns
        -------
        returns self.centroids which is None
        """
        print("DBSCAN cannot find clusters' centroids.")
        return self.centroids

    def print_info(self):
        """
        Reports the parameters used for the clustering.
        """
        print("\n#################")
        print("Running DBSCAN Algorithm with parameters:")
        print(f"  - eps \t\t {self.eps}")
        print(f"  - metric \t\t {self.metric}")
        print(f"    - metric_param \t {self.metric_param}")
        print(f"  - min_samples \t {self.min_samples}")
        if self.n_jobs is None:
            print(f"  - n_jobs \t\t 1")
        else:
            print(f"  - n_jobs \t\t {self.n_jobs}")
        if self.data_weight is None:
            print("  - Weights for each pose were not specified, so they all have the same weight.")
        else:
            print("  - Using weights for each pose.")
        print("#################\n")
