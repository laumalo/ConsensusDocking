"""
Clustering module
"""
import os
import pandas as pd
from parserEncoding import ParserEncoding


class Clustering:
    def __init__(self, encoding_file_path, clustering_method, metric='euclidean', eps=None, xi=None, metric_param=None,
                 min_samples=5, use_coord=True, use_norm_sc=False, data_sel_list=None, data_weight=None, n_jobs=None,
                 n_clusters=8, initial_clusters='k-means++', max_iter=300, n_init=10):
        """
        It initializes a specific Parser object for each clustering method.
        Parameters
        ----------
        encoding_file_path : str
            Path to encoding file. This file must contain the data that wants to be used for the clustering.
        clustering_method : str
            Clustering method that you want to use for the clustering.
        metric : str
            Metric used to compute distances between clusters for DBSCAN and OPTICS algorithms.
        eps : float
            The maximum distance between two samples for them to be considered as in the same neighborhood. Used only 
            when clustering_method='dbscan'.
        xi : float between 0 and 1
            Determines the minimum steepness on the reachability plot that constitutes a cluster boundary. For example, 
            an upwards point in the reachability plot is defined by the ratio from one point to its successor being at 
            most 1-xi. Used only when clustering_method='optics'.
        metric_param : array_like for mahalanobis/ int for minkowski / 1D array_like for seuclidean / otherwise None
            Extra parameter needed for some metrics.
            Mahalanobis:
                The inverse of the covariance matrix.
            Minkowski:
                When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2.
            Seuclidean:
                 1-D array of component variances. It is usually computed among a larger collection vectors.
        min_samples : int
            The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
             This includes the point itself.
        use_coord : bool
            If True, it will select the 9 coordinates in the encoding file to do the clustering. If False, you must 
            specify which columns do you want to use in the data_sel_list parameter. Notice that if data_sel_list is 
            specified, the data_sel_list columns will have priority (so use_coord will be ignored)
        use_norm_sc : bool
            If False, it won't use the norm_score date in the encoding file to cluster. If True, it will select the 
            norm_score info in the encoding file to cluster the data, however you must also set use_coord=True, 
            otherwise you will obtain a ValueError (since we consider that only using the norm_score is not enough to 
            make a good clustering).
        data_sel_list : list of str
            List containing the names of the columns that we want to select from the encoding file to make the
             clustering. Note that, when specifying this parameter, it will be used regardless the use_coord and 
             use_norm_sc parameters. 
        data_weight : array-like of shape (n_samples,)
            Array like object specifying the weight that we want to use for each pose (taking into account the order in
            the encoding file). Remember scaling min_samples accordingly.
            Note: Ideally this parameter will end up being a bool parameter since the parserEncoding object will be able
             to generate a list of weight of each pose according the program that generate that pose.
        n_jobs : int
            The number of parallel jobs to run. None means 1. -1 means using all processors.
        n_clusters : int
            The number of clusters to form as well as the number of centroids to generate with KMeans algorithm.
        initial_clusters : str or array (n_cluster, n_features)
            Only used with KMeans algorithm.
            ‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
            ‘random’: choose n_clusters observations (rows) at random from data for the initial centroids.
            If an array is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.
        max_iter : int
            Maximum number of iterations of the k-means algorithm for a single run. Only used with KMeans algorithm.
        n_init : int
            Number of time the k-means algorithm will be run with different centroid seeds. The final results will be
            the best output of n_init consecutive runs in terms of inertia. Only used with KMeans algorithm.
        """
        self.encoding_file_path = encoding_file_path
        self.clustering_method = clustering_method.lower()
        self.metric = metric.lower()
        self.metric_param = metric_param
        self.min_samples = min_samples
        self.labels = None
        self.data_weight = data_weight

        if data_sel_list is not None:
            print(f"Using the following columns: {data_sel_list}.")
            self.data = self.__get_selected_col(data_sel_list)
        elif use_coord and not use_norm_sc:
            print("Only using coordinates.")
            self.data = self.__get_coord()
        elif use_coord and use_norm_sc:
            print("Using coordinates and normalized scores.")
            self.data = self.__get_coord_norm_sc()
        elif use_norm_sc and not use_coord:
            raise ValueError('You cannot only use norm_score to cluster! Please, specify which columns do you want to '
                             'select with the input parameter data_sel_list or select use_coord=True to also use '
                             'coordinates')
        elif not use_norm_sc and not use_coord and data_sel_list is None:
            raise ValueError('At least use_coord must be True to have data to make the clustering!')

        if self.clustering_method == 'dbscan':
            from clusteringDBSCAN import ClusteringDBSCAN
            self.model = ClusteringDBSCAN(self.data, eps=eps, metric=self.metric, metric_param=self.metric_param,
                                          min_samples=self.min_samples, data_weight=self.data_weight, n_jobs=n_jobs)
        elif self.clustering_method == 'optics':
            from clusteringOPTICS import ClusteringOPTICS
            self.model = ClusteringOPTICS(self.data, metric=self.metric, metric_param=self.metric_param,
                                          cluster_method='xi', xi=xi, eps=eps, min_samples=self.min_samples,
                                          n_jobs=n_jobs)
        elif self.clustering_method == 'kmeans':
            from clusteringKMeans import ClusteringKMeans
            self.model = ClusteringKMeans(self.data, n_clusters=n_clusters, initial_clusters=initial_clusters,
                                          max_iter=max_iter, n_init=n_init, data_weight=self.data_weight)
            if n_jobs is not None and n_jobs > 1:
                print(f"Warning: Despite specifying n_jobs to {n_jobs}, KMeans clustering can only run as a single job")

    def __get_coord(self):
        """
        Parses the encoding file and gets coord using a ParserEncoding object and returns it in a DataFrame.
        Returns
        -------
        DataFrame with the nine coordinates in the encoding file.
        """
        parser = ParserEncoding(self.encoding_file_path)
        return parser.get_coord()

    def __get_coord_norm_sc(self):
        """
        Parses the encoding file and gets coord and normalized score using a ParserEncoding object and returns it in a
        DataFrame.
        Returns
        -------
        DataFrame with the nine coordinates and the normalized score in the encoding file.
        """
        parser = ParserEncoding(self.encoding_file_path)
        return parser.get_coord_norm_sc()

    def __get_selected_col(self, selected_columns):
        """
        Parses the encoding file and gets the input columns using a ParserEncoding object and returns it in a
        DataFrame.
        Parameters
        ----------
        selected_columns : List of str
            List containing the names of the columns that we want to select from the encoding file.

        Returns
        -------
        DataFrame with the selected columns information in the encoding file.
        """
        parser = ParserEncoding(self.encoding_file_path)
        return parser.get_columns_by_name(selected_columns)

    def __get_ids_by_index(self, index_list):
        """
        Parses the encoding file and gets the ids of the poses given the index (ie. row number).
        Parameters
        ----------
        index_list : List of int
            List of indexes corresponding to the poses we want to get the ids.
        Returns
        -------
        List of str with the ids for each pose.
        """
        parser = ParserEncoding(self.encoding_file_path)
        return parser.get_ids_by_row(index_list)

    def __get_cluster_counts(self):
        """
        Counts how many poses contains each cluster given the list of labels.
        Returns
        -------
        Sorted list (decreasing order) of tuples. Each tuple contains (cluster_id (int), cluster_size (int)).
        Cluster with label -1 are the outliers (not a cluster)
        """
        from collections import Counter
        return sorted([(a, b) for a, b in Counter(self.labels).items() if b > 1], key=lambda x: -x[1])

    def __cluster_report(self):
        """
        Prints a small report of the clustering
        """
        cluster_counts = self.__get_cluster_counts()
        if self.labels[self.labels == -1].size != 0:
            print(f' - Number of clusters: {len(cluster_counts) - 1}')
            print(f' - Outliers: {self.labels[self.labels == -1].size}')
        else:
            print(f' - Number of clusters: {len(cluster_counts)}')
            print(f' - Outliers: 0')
        valid_clusters = [cluster for cluster in cluster_counts if not cluster[0] == -1]
        print(' - Most populated clusters Top 10')
        for i, cluster in enumerate(valid_clusters[:10]):
            print(f'     {i + 1}. \t Cluster id: {cluster[0]} \t Number of poses: {cluster[1]}')

    def __get_cluster_index(self, save_dict_to_yaml=False, yaml_path='.'):
        """
        Obtains all the poses indexes for all the clusters and save them in a dict object that it's returned. If
        save_dict_to_yaml=True, then the dict will be saved in a yaml file, so it can be loaded later.
        Parameters
        ----------
        save_dict_to_yaml : bool
            If True it will save the cluster_index_dict to a file
        yaml_path : str
            Path to the folder in which we want to save the yaml file (if save_dict_to_yaml=True). Notice that the
            output file will be always have the filename: cluster_index_dict.yaml.
        Returns
        -------
        Dict with cluster id as keys and indexes of the poses in each cluster as values.
        """
        cluster_index_dict = {}
        for i, cluster in enumerate(self.labels):
            if cluster not in cluster_index_dict.keys():
                cluster_index_dict[int(cluster)] = []
            cluster_index_dict[cluster].append(i)
        # self.cluster_index_dict = cluster_index_dict
        if save_dict_to_yaml:
            filename = f'{self.clustering_method}_cluster_index_dict.yaml'
            self.__write_yaml(cluster_index_dict, filename, yaml_path)
            print(f"Saved cluster index dict to {os.path.join(yaml_path, filename)}")
        return cluster_index_dict

    def get_cluster_poses(self, save_poses_dict_to_yaml=True, save_index_dict_to_yaml=False, yaml_path='.'):
        """
        Obtains all the poses ids for all the clusters and save them in a dict object that it's returned. If
        save_dict_to_yaml=True, then the dict will be saved in a yaml file, so it can be loaded later. If
        save_index_dict_to_yaml=True we will also save the dict with the indexes (rows in the input encoding file).
        Parameters
        ----------
        save_poses_dict_to_yaml : bool
            If True it will save a yaml file containing the poses ids that belong to each cluster.
            Example: {cluster_id1: [pose1,pose2...], cluster_id2: [pose3,pose4],...}
        save_index_dict_to_yaml : bool
            If True it will save a yaml file of the poses index that belong to each cluster.
            Example: {cluster_id1: [ind1,ind2...], cluster_id2: [ind3,ind4],...}
        yaml_path : str
            Path to the folder in which we want to save the yaml file (if save_dict_to_yaml=True). Notice that the
            output file will be always have the filename: cluster_poses_dict.yaml.

        Returns
        -------
        Dict with cluster id as keys and poses ids of each cluster as values.
        """
        cluster_pose_dict = {}
        cluster_index_dict = self.__get_cluster_index(save_dict_to_yaml=save_index_dict_to_yaml, yaml_path=yaml_path)
        for cluster, index_list in cluster_index_dict.items():
            poses = self.__get_ids_by_index(index_list)
            cluster_pose_dict[int(cluster)] = poses
        if save_poses_dict_to_yaml:
            filename = f'{self.clustering_method}_cluster_poses_dict.yaml'
            self.__write_yaml(cluster_pose_dict, filename, yaml_path)
            print(f"Saved cluster poses dict to {os.path.join(yaml_path, filename)}")
        return cluster_pose_dict

    def get_centroids_poses(self, save_centroid_poses_to_yaml=True, save_centroid_index_to_yaml=False, yaml_path='.'):
        """
        Obtains the poses ids for the centroids of each cluster and save them in a dict object that it's returned. If
        save_dict_to_yaml=True, then the dict will be saved in a yaml file, so it can be loaded later. If
        save_centroid_index_to_yaml=True, we will also save the dict with the indexes (rows in the input encoding file).
        Parameters
        ----------
        save_centroid_poses_to_yaml : bool
            If True it will save a yaml file containing the poses ids that of each cluster centroid.
            Example: {cluster_id1: centroid_pose1, cluster_id2: centroid_pose2,...}
        save_centroid_index_to_yaml : bool
            If True it will save a yaml file of the centroid poses index that belong to each cluster.
            Example: {cluster_id1: centroid_ind1, cluster_id2: centroid_ind2,...}
        yaml_path : str
            Path to the folder in which we want to save the yaml file (if save_dict_to_yaml=True). Notice that the
            output file will be always have the filename: cluster_poses_dict.yaml.
        Returns
        -------
        Dict with cluster id as keys and poses ids of each cluster as values.
        """
        centroids_index = self.model.get_centroids()
        centroid_pose_dict = {}
        centroid_index_dict = {}
        if centroids_index is not None:
            for cluster, index in enumerate(centroids_index):
                pose = self.__get_ids_by_index([index])
                centroid_pose_dict[int(cluster)] = pose[0]
                centroid_index_dict[int(cluster)] = int(index)
        if save_centroid_poses_to_yaml:
            filename = f'{self.clustering_method}_centroid_poses_dict.yaml'
            self.__write_yaml(centroid_pose_dict, filename, yaml_path)
            print(f"Saved cluster poses dict to {os.path.join(yaml_path, filename)}")
        if save_centroid_index_to_yaml:
            filename = f'{self.clustering_method}_centroid_index_dict.yaml'
            self.__write_yaml(centroid_index_dict, filename, yaml_path)
            print(f"Saved cluster poses dict to {os.path.join(yaml_path, filename)}")
        return centroid_pose_dict

    @staticmethod
    def __write_yaml(dictionary, filename, save_path='.'):
        """
        Dict with structure like:
        {0:[value1,value2], 1: [value3, value4], ...}
        """
        import yaml
        with open(os.path.join(save_path, filename), 'w') as f:
            yaml.dump(dictionary, f)

    def run(self, save_index_dict=False, save_poses_dict=False, save_centroid_poses=False, save_centroid_index=False,
            save_path='.'):
        """
        Runs the clustering, save the labels to self.labels, prints a report. If specified it can generate yaml files
        with the index and/or poses ids for each cluster.
        Parameters
        -------
        save_index_dict : bool
            If True, it saves a dict of the poses index that belong to each cluster.
        save_poses_dict : bool
            If True, it saves a dict of the poses ids that belong to each cluster.
        save_centroid_index : bool
            If True, it saves a yaml file of the pose index of each cluster centroid.
        save_centroid_poses : bool
            If True, it saves a yaml file of the poses ids of each cluster centroid.
        save_path : str
            Path to the folder in which we want to save the yaml file (if save_dict_to_yaml=True). Notice that the
            output file will be always have the filename: cluster_poses_dict.yaml.
        """
        self.model.print_info()
        self.model.fit()
        self.labels = self.model.get_labels()
        self.__cluster_report()

        if save_poses_dict:
            if save_index_dict:
                self.get_cluster_poses(save_index_dict_to_yaml=True, yaml_path=save_path)
            else:
                self.get_cluster_poses(yaml_path=save_path)
        elif save_index_dict:
            self.__get_cluster_index(save_dict_to_yaml=True, yaml_path=save_path)

        if save_centroid_poses:
            if save_centroid_index:
                self.get_centroids_poses(save_centroid_index_to_yaml=True, yaml_path=save_path)
            else:
                self.get_centroids_poses(yaml_path=save_path)
        elif save_centroid_index:
            self.get_centroids_poses(save_centroid_poses_to_yaml=False, save_centroid_index_to_yaml=True,
                                     yaml_path=save_path)
