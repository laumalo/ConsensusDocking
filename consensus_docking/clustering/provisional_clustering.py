import os
import numpy as np
import pandas as pd
import scipy.spatial as spatial
from collections import Counter, defaultdict
import linecache
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances_argmin_min

import argparse as ap
import logging
import sys

logging.basicConfig(
    format='[%(module)s] - %(levelname)s: %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.INFO, stream=sys.stdout)


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
    parser = ap.ArgumentParser(description="Clustering algorithm.")
    parser.add_argument("encoding_file", type=str,
                        help="Encoding file")
    parser.add_argument("-n", "--n_clusters", type=int,
                        help="Number of clusters to generate.", default=30)
    parsed_args = parser.parse_args()
    return parsed_args

def get_weighted_representatives(n_clusters):
    if n_clusters % 15 == 0: 
        step = int(n_clusters/15)
        representatives_weighted = \
            list(reversed([x + step for x in range(0,5*step, step)]))
    else: 
        residue = n_clusters%15
        step = int((n_clusters-residue)/15)
        representatives_weighted = \
            list(reversed([x + step for x in range(0,5*step, step)]))
        add = int(residue/5)
        representatives_weighted = [x + add for x in representatives_weighted]
        if not residue%5 == 0: 
            res_add = residue%5
            representatives_weighted[0] = representatives_weighted[0]+res_add
    return representatives_weighted

def clustering(encoding_file, eps_dbscan = 6):
    df = pd.read_csv(encoding_file)
    df.columns= \
    ['Unnamed: 0','File', 'Score','x1','y1','z1','x2','y2','z2','x3','y3','z3']
    df_coords = df[['x1','y1','z1','x2','y2','z2','x3','y3','z3']]


    model = DBSCAN(eps=eps_dbscan, metric='euclidean')
    model.fit(df_coords.values)
    top_clusters = sorted([(a,b) for a,b in Counter(model.labels_).items() 
                   if b>1], key=lambda x: -x[1])

    file_names = [name[0] for name in df[['File']].values]
    d1 = {}
    for file_name, label in zip(file_names, model.labels_): 
        d1[file_name] = label

    logging.info(' - Number of clusters: {}'.format(len(top_clusters)))
    logging.info(' - Outliers: {}'
        .format(model.labels_[model.labels_ == -1].size))
    valid_clusters = \
        [cluster for cluster in top_clusters if not cluster[0] == -1]
    logging.info(' - Top 5 DBSCAN clusters')
    selected_labels = []
    for i,cluster in enumerate(valid_clusters[:5]):
        logging.info('     Cluster {} \t Label {} \t Number of poses: {}'
            .format(i + 1, cluster[0], cluster[1]))
        selected_labels.append(cluster[0])
    return d1, selected_labels

def main(args):
    
    # BSCAN clustering
    d_clusters, selected_labels = clustering(args.encoding_file)
    df = pd.read_csv(args.encoding_file)

    logging.info(
        ' - PDBs of top {} most representative clusters with DBSCAN+K-Means.'
        .format(args.n_clusters))
    
    # Get number of kmeans clusters
    representative_clusters_num = get_weighted_representatives(args.n_clusters)
    representative_structures =  []
    
    # K-Means clustering
    d_clustering = {}
    for label, num_poses in zip(selected_labels, representative_clusters_num):
        poses_cluster_label = [k for k,v in d_clusters.items() if v == label]

        df_filtered = df[df['File'].isin(poses_cluster_label)]
        df_coords_cluster = \
            df_filtered[['x1','y1','z1','x2','y2','z2','x3','y3','z3']]

        model = KMeans(n_clusters=num_poses)
        model.fit(df_coords_cluster.values)
        clusters = list(model.labels_)
        top_clusters = sorted([(a,b) for a,b in Counter(model.labels_).items() 
                       if b>1], key=lambda x: -x[1])

        file_names = [name[0] for name in df_filtered[['File']].values]
        
        d_kmeans = defaultdict(list)
        for file_name, label_model in zip(file_names, model.labels_): 
            cluster_label = 'cluster_{}_{}'.format(label, label_model)
            d_kmeans[cluster_label].append(os.path.basename(file_name))


        names = df_filtered[['File']].values
        coords = \
            df_filtered[['x1','y1','z1','x2','y2','z2','x3','y3','z3']].values

        d_coods_names = {name[0] : coord for coord,name in zip(coords, names)}
    
        centroids = model.cluster_centers_
        closest, _ = pairwise_distances_argmin_min(model.cluster_centers_,
                                                   df_coords_cluster.values)
        

        
        for idx in closest:
            coord_centroid = df_coords_cluster.values[idx]
            for k,v in d_coods_names.items(): 
                if np.all(v == coord_centroid):
                    representative_structures.append(os.path.basename(k))

        d_clustering = {**d_clustering, **d_kmeans}
    
    # Results dictionary
    d_results = {}
    for rep in representative_structures: 
        cluster_label = [k for k,v in d_clustering.items() if rep in list(v)][0]
        d_results[cluster_label] = \
            {'Representative': rep,
             'Population' : len(d_clustering[cluster_label]),
             'Structures' : d_clustering[cluster_label]}


    # Consensus
    NUMBER_OF_SOFTWARES = 5 
    consensus_dump = []
    for k,v in d_results.items(): 
        
        labels = list([x.split('_')[0] for x in v['Structures']])
        consensus_count = dict(Counter(map(str,labels)))
        if len(consensus_count.keys()) == NUMBER_OF_SOFTWARES: 
            logging.info(f"     - {k} \t Representative:{v['Representative']}" +  
                         f" \t Population: {v['Population']}.")
        else: 
            consensus_dump.append(k)
    print('Clusters not fulfilling the consensus criteria:')
    for k in consensus_dump: 
        logging.info(f"     - {k} \t Representative:{v['Representative']}" +  
                     f" \t Population: {v['Population']}.")



if __name__ == '__main__':
    args = parse_args()
    main(args)