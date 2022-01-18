import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS


class ClusteringOPTICS:
    def __init__(self, df, metric='minkowski', metric_param=None, cluster_method='xi', xi=None, eps=None,
                 min_samples=5, n_jobs=None):
        """
        df : Pandas DataFrame
            DataFrame containing the data to be used for the clustering.
            Shape (n_samples, n_features)
        metric : str
            The metric to use when calculating distance between instances in a feature array.
            It can be: [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’] and [‘braycurtis’, ‘canberra’,
            ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘minkowski’,
            ‘rogerstanimoto’, ‘russellrao’,‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]
        metric_param : array_like for mahalanobis/ int for minkowski / 1D array_like for seuclidean / otherwise None
            Extraparameter needed for some metrics.
            Mahalanobis:
                The inverse of the covariance matrix.
            Minkowski:
                When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2.
            Seuclidean:
                 1-D array of component variances. It is usually computed among a larger collection vectors.
        cluster_method : str
            The extraction method used to extract clusters using the calculated reachability and ordering. Possible values are “xi” and “dbscan”.
        eps : float
            The maximum distance between two samples for them to be considered as in the same
            neighborhood. Used only when cluster_method='dbscan'.
        xi : float between 0 and 1
            Determines the minimum steepness on the reachability plot that constitutes a cluster
            boundary. For example, an upwards point in the reachability plot is defined by the
            ratio from one point to its successor being at most 1-xi.
            Used only when cluster_method='xi'.
        min_samples : int
            The number of samples (or total weight) in a neighborhood for a point to be
            considered as a core point. This includes the point itself.
        n_jobs: int
            The number of parallel jobs to run. None means 1. -1 means using all processors.
        """
        self.data = df
        self.cluster_method = cluster_method.lower()
        self.eps = eps  # epsilon parameter in DBSCAN algorithm
        self.xi = xi  # xi parameter in Xi algorithm
        self.metric = metric.lower()
        self.min_samples = min_samples
        self.labels = None

        self.__validate_input()

        if metric == 'minkowski':
            assert metric_param is not None, "metric_param should be an int value: p = 1, this is equivalent to using" \
                                             " manhattan_distance (l1), and euclidean_distance (l2) for p = 2."
            self.model = OPTICS(min_samples=self.min_samples, xi=self.xi, eps=self.eps, metric=self.metric,
                                p=metric_param, n_jobs=n_jobs)
        elif metric == 'mahalanobis':
            raise ModuleNotFoundError("Still not implemented. Try another metric.")
        elif metric == 'seuclidean':
            raise ModuleNotFoundError("Still not implemented. Try another metric.")
        else:
            self.model = OPTICS(min_samples=self.min_samples, xi=self.xi, eps=self.eps, metric=self.metric, n_jobs=n_jobs)

    def __validate_input(self):
        """
        Verifies if the inputs are coherent, otherwise it will raise an error or print a warning.
        """
        valid_metrics = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan', 'braycurtis', 'canberra',
                         'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
                         'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
                         'sqeuclidean', 'yule']
        valid_cluster_methods = ['dbscan', 'xi']

        assert self.metric in valid_metrics, f"{self.metric} is not available, try with any of the following:" \
                                             f" {valid_metrics}."
        assert self.cluster_method in valid_cluster_methods, f"{self.cluster_method} is invalid. Select one of the" \
                                                             f" followings: {valid_cluster_methods}."

        if self.cluster_method == 'xi':
            assert self.xi is not None, "You selected xi as clustering method, so you need to define xi parameter" \
                                        " (Sklearn default = 0.05)."
        elif self.cluster_method == 'dbscan':
            assert self.eps is not None, "You selected dbscan as clustering method, so you need to define eps parameter."
            # When xi is None, we get: TypeError: unsupported operand type(s) for -: 'int' and 'NoneType'
            self.xi = 0.05  # To avoid TypeError. 0.05 is the default value in sklearn for xi.

    def fit(self):
        """
        Fits the model with self.data and, The fitted model is returned and saved in self.model.
        Returns
        -------
        OPTICS scklearn object model fitted.
        """
        self.model.fit(self.data.values)
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
