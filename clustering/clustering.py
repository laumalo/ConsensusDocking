"""
Clustering module
"""
import os
import pandas as pd
from parserEncoding import ParserEncoding


class Clustering:
    def __init__(self, encoding_file_path, clustering_method, metric='euclidean', eps=None, xi=None, metric_param=None,
                 min_samples=5, use_coord=True, use_norm_sc=False, data_sel_list=None, data_weight=None, n_jobs=None):
        """
        It initializes a specific Parser object for each clustering method.
        Parameters
        ----------
        encoding_file_path : str
            Path to encoding file. This file must contain the data that wants to be used for the clustering.
        clustering_method : str
            Clustering method that you want to use for the clustering.
        metric : str
            Metric you want to use to compute distances between clusters.
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
        data_weight : list
            List of specifying the weight that we want to use for each pose (taking into account the order in the
             encoding file). Remember scaling min_samples accordingly.
            Note: Ideally this parameter will end up being a bool parameter since the parserEncoding object will be able
             to generate a list of weight of each pose according the program that generate that pose.
        n_jobs : int
            The number of parallel jobs to run. None means 1. -1 means using all processors.
        """
        self.encoding_file_path = encoding_file_path
        self.clustering_method = clustering_method.lower()
        self.metric = metric.lower()
        self.eps = eps
        self.xi = xi
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
            print("Using coordinates and normalized score.")
            self.data = self.__get_coord_norm_sc()
        elif use_norm_sc and not use_coord:
            raise ValueError('You cannot only use norm_score to cluster! Please, specify which columns do you want to '
                             'select with the input parameter data_sel_list or select use_coord=True to also use '
                             'coordinates')
        elif not use_norm_sc and not use_coord and data_sel_list is None:
            raise ValueError('At least use_coord must be True to have data to make the clustering!')

        if self.clustering_method == 'dbscan':
            from clusteringDBSCAN import ClusteringDBSCAN
            self.model = ClusteringDBSCAN(self.data, eps=self.eps, metric=self.metric, metric_param=self.metric_param,
                                          min_samples=self.min_samples, data_weight=self.data_weight, n_jobs=n_jobs)
        elif self.clustering_method == 'optics':
            from clusteringOPTICS import ClusteringOPTICS
            self.model = ClusteringOPTICS(self.data, metric=self.metric, metric_param=self.metric_param,
                                          cluster_method='xi', xi=self.xi, eps=self.eps, min_samples=self.min_samples,
                                          n_jobs=n_jobs)
        elif self.clustering_method == 'kmeans':
            raise ModuleNotFoundError("Kmeans module is on construction.")
            pass

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
            self.__write_yaml(cluster_index_dict, 'cluster_index_dict.yaml', yaml_path)
            print(f"Saved cluster index dict to {os.path.join(yaml_path, 'cluster_index_dict.yaml')}")
        return cluster_index_dict

    def get_cluster_poses(self, save_poses_dict_to_yaml=True, save_index_dict_to_yaml=False, yaml_path='.'):
        """
        Obtains all the poses ids for all the clusters and save them in a dict object that it's returned. If
        save_dict_to_yaml=True, then the dict will be saved in a yaml file, so it can be loaded later.
        Parameters
        ----------
        save_poses_dict_to_yaml : bool
            If True it will save the cluster_index_dict to a yaml file.
        save_index_dict_to_yaml : bool
            If True it will save  a dict of the poses index that belong to each cluster.
        yaml_path : str
            Path to the folder in which we want to save the yaml file (if save_dict_to_yaml=True). Notice that the
            output file will be always have the filename: cluster_poses_dict.yaml.

        Returns
        -------
        Dict with cluster id as keys and poses ids of each cluster as values.
        """
        cluster_pose_dict = {}
        cluster_index_dict = self.__get_cluster_index(save_dict_to_yaml=save_index_dict_to_yaml)
        for cluster, index_list in cluster_index_dict.items():
            poses = self.__get_ids_by_index(index_list)
            cluster_pose_dict[int(cluster)] = poses
        if save_poses_dict_to_yaml:
            self.__write_yaml(cluster_pose_dict, 'cluster_poses_dict.yaml', yaml_path)
            print(f"Saved cluster poses dict to {os.path.join(yaml_path, 'cluster_poses_dict.yaml')}")
        return cluster_pose_dict

    @staticmethod
    def __write_yaml(dictionary, filename, save_path='.'):
        """
        Dict with structure like:
        {0:[value1,value2], 1: [value3, value4], ...}
        """
        import yaml
        with open(os.path.join(save_path, filename), 'w') as f:
            yaml.dump(dictionary, f)

    def run(self, save_index_dict=False, save_poses_dict=False, save_path='.'):
        """
        Runs the clustering, save the labels to self.labels, prints a report. If specified it can generate yaml files
        with the index and/or poses ids for each cluster.
        Parameters
        -------
        save_index_dict : bool
            If True, it saves a dict of the poses index that belong to each cluster.
        save_poses_dict : bool
            If True, it saves a dict of the poses ids that belong to each cluster.
        save_path : str
            Path to the folder in which we want to save the yaml file (if save_dict_to_yaml=True). Notice that the
            output file will be always have the filename: cluster_poses_dict.yaml.
        """
        self.model.fit()
        self.labels = self.model.get_labels()
        self.__cluster_report()

        if save_poses_dict:
            print('in3')
            if save_index_dict:
                self.get_cluster_poses(save_index_dict_to_yaml=True, yaml_path=save_path)
            else:
                self.get_cluster_poses(save_index_dict_to_yaml=False, yaml_path=save_path)
        elif save_index_dict:
            print('in')
            self.__get_cluster_index(save_dict_to_yaml=True, yaml_path=save_path)


