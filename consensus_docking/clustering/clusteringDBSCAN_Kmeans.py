import os
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances_argmin_min

import logging
import sys

logging.basicConfig(
    format='[%(module)s] - %(levelname)s: %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.INFO, stream=sys.stdout)


class TwoStepsClustering(object): 
    """
    It initializates a TwoStepClustering class.
    """
    def __init__(self, encoding_file, n_clusters,
                 eps_DBSCAN = 6, metric_DBSCAN = 'euclidean'): 
        self.encoding_file = encoding_file
        self.n_clusters = n_clusters 
        self.eps_DBSCAN = eps_DBSCAN
        self.metric_DBSCAN = metric_DBSCAN


    def get_coords_names(self): 
        """
        It gets the coordinates and the file names from the encoding file. 

        Returns
        -------
        df_coords : pd.Dataframe
            Dataframe containing all the coordinates. 
        file_names : list
            List of all the file names. 
        """
        df = pd.read_csv(self.encoding_file)
        df.columns= ['File', 'Score','x1','y1','z1',
                     'x2','y2','z2','x3','y3','z3']
        df_coords = df[['x1','y1','z1','x2','y2','z2','x3','y3','z3']]
        file_names = [name[0] for name in df[['File']].values]
        return df_coords, file_names

    def get_weighted_representatives(self, n_clusters):
        """
        It gets the number of K-means clusters to obtain from each DBSCAN 
        cluster.

        Parameters
        ----------
        n_clusters : int
            Number of total clusters to obtain. 

        Returns
        -------
        representatives_weighted : list
            Number of K-means clusters for each DBSCAN cluster. 
        """
        
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
            representatives_weighted = \
                [x + add for x in representatives_weighted]
            if not residue%5 == 0: 
                res_add = residue%5
                representatives_weighted[0] = \
                    representatives_weighted[0]+res_add
        return representatives_weighted

    def DBSCAN_step(self, coords, file_names, DBSCAN_CLUSTERS = 5):
        """
        DBSCAN: first clustering step. 

        Parameters
        ----------
        coords : pd.Dataframe
            Dataframe containing all the coordinates.
        file_names : list
            List of all the file names. 
        DBSCAN_CLUSTERS : int
            Number of clusters to keep for the next clustering step. Default: 5. 

        Returns
        -------
        d_clusters : dict
            Dictionary with the results of the DBSCAN clustering. 
        selected_labels : list
            List of cluster labels selected for the next clustering step.
        """
        
        model = DBSCAN(eps=self.eps_DBSCAN, metric=self.metric_DBSCAN)
        model.fit(coords.values)
        top_clusters = sorted([(a,b) for a,b in Counter(model.labels_).items() 
                       if b>1], key=lambda x: -x[1])

        d_clusters = {}
        for file_name, label in zip(file_names, model.labels_): 
            d_clusters[file_name] = label

        logging.info(' - Number of clusters: {}'.format(len(top_clusters)))
        logging.info(' - Outliers: {}'
            .format(model.labels_[model.labels_ == -1].size))
        valid_clusters = \
            [cluster for cluster in top_clusters if not cluster[0] == -1]
        logging.info(' - Top 5 DBSCAN clusters')
        selected_labels = []
        for i,cluster in enumerate(valid_clusters[:DBSCAN_CLUSTERS]):
            logging.info('     Cluster {} \t Label {} \t Number of poses: {}'
                .format(i + 1, cluster[0], cluster[1]))
            selected_labels.append(cluster[0])
        return d_clusters, selected_labels

    def KMeans_step(self, d_clusters, selected_labels): 
        """
        K-Means : second clustering step

        Parameters
        ----------
        d_clusters : dict
            Dictionary with the results of the DBSCAN clustering. 
        selected_labels : list
            List of cluster labels selected from DBSCAN clustering step.

        Returns
        -------
        d_clustering : dict
            Dictionary with the results of the K-Means clustering.
        representative_structures : list
            List of representative structures selected. 
        """
        representative_clusters= \
            self.get_weighted_representatives(self.n_clusters)
        
        df = pd.read_csv(self.encoding_file)
        d_clustering = {}
        representative_structures = []
        for label, num_poses in zip(selected_labels, representative_clusters):
            poses_cluster_label = \
                [k for k,v in d_clusters.items() if v == label]

            df_filtered = df[df['File'].isin(poses_cluster_label)]
            df_coords_cluster = \
                df_filtered[['x1','y1','z1','x2','y2','z2','x3','y3','z3']]

            model = KMeans(n_clusters=num_poses)
            model.fit(df_coords_cluster.values)
            clusters = list(model.labels_)
            file_names = [name[0] for name in df_filtered[['File']].values]
            
            d_kmeans = defaultdict(list)
            for file_name, label_model in zip(file_names, model.labels_): 
                cluster_label = 'cluster_{}_{}'.format(label, label_model)
                d_kmeans[cluster_label].append(os.path.basename(file_name))

            names = df_filtered[['File']].values
            coords = df_filtered[
                        ['x1','y1','z1','x2','y2','z2','x3','y3','z3']].values

            d_coods_names = \
                {name[0] : coord for coord,name in zip(coords, names)}

            centroids = model.cluster_centers_
            closest, _ = \
                pairwise_distances_argmin_min(model.cluster_centers_,
                                              df_coords_cluster.values)
            
            
            for idx in closest:
                coord_centroid = df_coords_cluster.values[idx]
                for k,v in d_coods_names.items(): 
                    if np.all(v == coord_centroid):
                        representative_structures.append(os.path.basename(k))
            d_clustering = {**d_clustering, **d_kmeans}

        return d_clustering, representative_structures


    def apply_consensus(self, d, NUMBER_OF_SOFTWARES = 5): 
        """
        It applies the consensus criteria to prioritize clusters with structures
        from all the softwares. 

        Parameters
        ---------- 
        d : dict
            Dictionary with all the clustering results. 
        NUMBER_OF_SOFTWARES : int
            Total number of docking softwares used. Default: 5. 
        """
        consensus_dump = []
        for k,v in d.items(): 
            labels = list([x.split('_')[0] for x in v['Structures']])
            consensus_count = dict(Counter(map(str,labels)))
            if len(consensus_count.keys()) == NUMBER_OF_SOFTWARES: 
                logging.info(
                    f"     - {k} \t Representative:{v['Representative']}" +  
                    f" \t Population: {v['Population']}.")
            else: 
                consensus_dump.append(k)
        logging.info('- Clusters not fulfilling the consensus criteria:')
        
        for k in consensus_dump: 
            logging.info(
                f"     - {k} \t Representative:{d.get(k)['Representative']}" +  
                f" \t Population: {d.get(k)['Population']}.")

    def run(self): 
        """
        It runs the clustering pipeline based on the DBSCAN-Kmeans two-step 
        clustering.
        """

        # DBSCAN clustering
        coords,file_names = self.get_coords_names()
        logging.info('STEP ONE: DBSCAN clustering')
        d_dbscan, selected_labels = self.DBSCAN_step(coords, file_names)
        
        # K-Means clustering
        logging.info('STEP TWO: K-MEANS clustering')
        d_clustering, representative_structures = \
            self.KMeans_step(d_dbscan, selected_labels)

        # Dictionary clustering results
        d_results = {}
        for structure in representative_structures: 
            cluster_label = \
                [k for k,v in d_clustering.items() if structure in list(v)][0]
            d_results[cluster_label] = \
                {'Representative': structure,
                 'Population' : len(d_clustering[cluster_label]),
                 'Structures' : d_clustering[cluster_label]}

        # Consensus criteria
        self.apply_consensus(d_results)