import argparse as ap
import numpy as np
from clustering import Clustering


def parse_args():
    """
    It parses the command-line arguments.
    Parameters
    ----------
    args : list[str]
        List of command-line arguments to parse
    Returns
    -------
    parsed_args : argparse.Namespace
        It contains the command-line arguments that are supplied by the user
    """
    valid_metrics = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan', 'braycurtis', 'canberra',
                     'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
                     'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
                     'sqeuclidean', 'yule']
    parser = ap.ArgumentParser(description="Clusters the selected data of a given encoding file using the specified "
                                           "clustering method.")
    parser.add_argument("clustering_method", type=str, choices=['dbscan', 'optics', 'kmeans'],
                        help="Algorithm that we want to use for the clustering")
    parser.add_argument("encoding_file_path", type=str,
                        help="Path to the encoding file that wants to be clustered.")
    parser.add_argument("-m", "--metric", type=str, default='euclidean', choices=valid_metrics,
                        help="Metric used to compute distances between clusters for DBSCAN and OPTICS algorithms.")
    parser.add_argument("-mp", "--metric-param", default=None,
                        help="Extra parameter needed for some metrics. "
                             "Mahalanobis: The inverse of the covariance matrix. Minkowski: When mp = 1, this is "
                             "equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for mp = 2. "
                             "Seuclidean: 1-D array of component variances. It is usually computed among a larger "
                             "collection vectors.")
    parser.add_argument("-o", "--out-dir", type=str, default='.',
                        help="Path to folder where the yaml files will be saved when specified.")
    parser.add_argument("-scli", "--save-cluster-index", default=False, action='store_true',
                        help="If True, it saves a dict of the poses index that belong to each cluster to yaml file.")
    parser.add_argument("-sci", "--save-centroid-index", default=False, action="store_true",
                        help="If True, it saves a yaml file of the pose index of each cluster centroid.  Notice that"
                             "OPTICS and DBSCAN cannot find the centroids, so it will return a yaml file with an "
                             "empty dict.")
    parser.add_argument("-scp", "--save-centroid-poses", default=False, action="store_true",
                        help="If True, it saves a yaml file of the pose ids of each cluster centroid. Notice that "
                             "OPTICS and DBSCAN cannot find the centroids, so it will return a yaml file with an "
                             "empty dict.")

    parser.add_argument("-eps", "--epsilon", type=int, default=None,
                        help="The maximum distance between two samples for them to be considered as in the same "
                             "neighborhood. Used only when clustering_method is dbscan.")
    parser.add_argument("-xi", "--xi", type=float, default=None,
                        help="Float between 0 and 1 that Determines the minimum steepness on the reachability plot that"
                             " constitutes a cluster boundary. For example, an upwards point in the reachability plot "
                             "is defined by the ratio from one point to its successor being at most 1-xi. Used only "
                             "when clustering_method is optics.")
    parser.add_argument("-min-sam", "--min-samples", type=int, default=5,
                        help="The number of samples (or total weight) in a neighborhood for a point to be considered "
                             "as a core point. Only used in DBSCAN.")
    parser.add_argument("-nc", "--n-clusters", type=int, default=8,
                        help="The number of clusters to form as well as the number of centroids to generate with "
                             "KMeans algorithm. Only used for KMeans algorithm.")
    parser.add_argument("-ini-clust", "--initial-clusters", type=str, choices=['k-means++', 'random'],
                        default='k-means++',
                        help="‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to "
                             "speed up convergence. ‘random’: choose n_clusters observations (rows) at random from "
                             "data for the initial centroids. Only used with KMeans algorithm. Notice that we could "
                             "pass an array of shape (n_clusters, n_features) to set the initial centers, however this "
                             "option is not available using the command line.")
    parser.add_argument("-max-iter", "--max-iterations", type=int, default=300,
                        help="Maximum number of iterations of the k-means algorithm for a single run. Only used with "
                             "KMeans algorithm.")
    parser.add_argument("-n-init", "--n-init", type=int, default=10,
                        help="Number of time the k-means algorithm will be run with different centroid seeds. The final"
                             " results will be the best output of n_init consecutive runs in terms of inertia. Only "
                             "used with KMeans algorithm.")
    parser.add_argument("-nj", "--n-jobs", type=int, default=None,
                        help="The number of parallel jobs to run. None means 1. -1 means using all processors. It only "
                             "works for DBSCAN and OPTICS algorithms.")
    parser.add_argument("-dw", "--data-weight", type=str, default=None,
                        help="Comma-separated list (without whitespaces) of the weight of each pose in the encoding "
                             "(using the order of the encoding). If you add weights to each pose, check that "
                             "--min_samples is coherent.")

    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument("-c", '--use-coord', default=False, action='store_true',
                            help="Use coordinate data from the encoding for the clustering.")
    data_group.add_argument("-cs", '--use-coord-and-norm-scores', default=False, action='store_true',
                            help="Use coordinate and normalized scores data from the encoding for the clustering.")
    data_group.add_argument("-sel-col", "--select-columns", type=str, default=None,
                            help="Column names of the features in the encoding file that will be used for the "
                                 "clustering. If you want to select multiple columns, make coma-separated list "
                                 "(without whitespaces).")

    parsed_args = parser.parse_args()
    return parsed_args


def init_clustering(encoding_file_path, clustering_method, metric, eps, xi, metric_param, min_samples, use_coord,
                    use_norm_sc, data_sel_list, data_weight, n_jobs, n_clusters, initial_clusters, max_iter, n_init):
    """

    Parameters
    ----------
    encoding_file_path
    clustering_method
    metric
    eps
    xi
    metric_param
    min_samples
    use_coord
    use_norm_sc
    data_sel_list
    data_weight
    n_jobs
    n_clusters
    initial_clusters
    max_iter
    n_init

    Returns
    -------

    """
    clustering = Clustering(encoding_file_path, clustering_method, metric, eps, xi, metric_param, min_samples,
                            use_coord, use_norm_sc, data_sel_list, data_weight, n_jobs, n_clusters, initial_clusters,
                            max_iter, n_init)
    return clustering


def run_clustering(clustering, save_path, save_cluster_index, save_centroid_index, save_centroid_poses,
                   save_cluster_poses=True):
    """

    Parameters
    ----------
    clustering : Clustering object

    save_path
    save_cluster_index
    save_cluster_poses
    save_centroid_poses
    save_centroid_index

    Returns
    -------

    """
    clustering.run(save_index_dict=save_cluster_index, save_poses_dict=save_cluster_poses,
                   save_centroid_poses=save_centroid_poses, save_centroid_index=save_centroid_index,
                   save_path=save_path)
    return clustering

def main(args):
    """
    Calls parse_sc using command line arguments
    """
    if args.use_coord_and_norm_scores:
        args.use_coord = True
    if args.select_columns is not None:
        args.select_columns = args.select_columns.split(',')
    if args.metric_param is not None and args.metric == 'minkowski':
        args.metric_param = int(args.metric_param)

    if args.data_weight is not None:
        dummy = list(map(int, args.data_weight.split(',')))
        args.data_weight = np.array(dummy)

    model = init_clustering(args.encoding_file_path, args.clustering_method, args.metric, args.epsilon,
                            args.xi, args.metric_param, args.min_samples, args.use_coord, args.use_coord_and_norm_scores,
                            args.select_columns, args.data_weight, args.n_jobs, args.n_clusters, args.initial_clusters,
                            args.max_iterations, args.n_init)
    fitted_model = run_clustering(model, args.out_dir, args.save_cluster_index, args.save_centroid_index,
                                  args.save_centroid_poses)
    return fitted_model


if __name__ == '__main__':
    args = parse_args()
    main(args)
