"""
Clustering module
"""
import os
import numpy as np
import pandas as pd
import scipy.spatial as spatial
from sklearn.cluster import KMeans, spectral_clustering, AgglomerativeClustering, DBSCAN, OPTICS, Birch
from Bio.PDB import *
from tqdm import tqdm
from collections import Counter
from biopandas.pdb import PandasPdb
import linecache

class Clustering(): 
    def __init__(self, encoding_file, docking_program): 
        self.encoding_file = encoding_file
        self.docking_program = docking_program

        # Parsing
        self.df = self._parse_encoding_file()

    def _parse_encoding_file(self):
            df = pd.read_csv(self.encoding_file)
            df.columns=['x1','y1','z1','x2','y2','z2','x3','y3','z3']
            return df

    def _generate_clusters(self, model): 
        clusters = list(model.labels_)
        top_clusters = sorted([(a,b) for a,b in Counter(model.labels_).items() 
                   if b>1], key=lambda x: -x[1])

        labels = model.labels_
        d_clusters = {}
        file_names = [self.docking_program + f for f in os.listdir(self.docking_program) if f.endswith(".pdb")] 
        for k,n in top_clusters:
            indices = [i for i, v in enumerate(labels) if v==k]
            files = [file for i,file in enumerate(file_names) if i in indices]
        d_clusters[k] = files
        print(' - Number of clusters: {}'.format(len(top_clusters)))
        print(' - Outliers: {}'.format(model.labels_[model.labels_ == -1].size))
        valid_clusters = [cluster for cluster in top_clusters if not cluster[0] == -1]
        print(' - Top 10 clusters')
        for i,cluster in enumerate(valid_clusters[:10]):
            print('     Cluster {} \t Number of poses: {}'.format(i + 1, cluster[1]))


    def DBSCAN(self, eps = 3*8):
        model = DBSCAN(eps=eps, metric='euclidean')
        model.fit(self.df.values)

        self._generate_clusters(model)




