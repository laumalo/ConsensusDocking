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



def clustering(encoding_file, docking_program):
    df = pd.read_csv(encoding_file)
    df.columns=['x1','y1','z1','x2','y2','z2','x3','y3','z3']
    model = DBSCAN(eps=3*8, metric='euclidean')
    #model = KMeans(n_clusters=100)
    model.fit(df.values)
    clusters = list(model.labels_)
    top_clusters = sorted([(a,b) for a,b in Counter(model.labels_).items() 
                   if b>1], key=lambda x: -x[1])
    #print(top_clusters)
    labels = model.labels_
    d = {}
    
    file_names = [f'{docking_program}/' + f for f in os.listdir(f'{docking_program}/') if f.endswith(".pdb")] 
    for k,n in top_clusters:
        indices = [i for i, v in enumerate(labels) if v==k]
        files = [file for i,file in enumerate(file_names) if i in indices]
        d[k] = files
    print(' - Number of clusters: {}'.format(len(top_clusters)))
    print(' - Outliers: {}'.format(model.labels_[model.labels_ == -1].size))
    valid_clusters = [cluster for cluster in top_clusters if not cluster[0] == -1]
    print(' - Top 10 clusters')
    for i,cluster in enumerate(valid_clusters[:10]):
        print('     Cluster {} \t Number of poses: {}'.format(i + 1, cluster[1]))

encoding_file = 'result_ftdock_pdbs.csv'
docking_program = 'ftdock_pdbs'
clustering(encoding_file, docking_program)

