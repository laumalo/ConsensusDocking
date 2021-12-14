# General imports
import argparse as ap
from multiprocessing import Pool, Array
from functools import partial
import os
import numpy as np
import pandas as pd
import scipy.spatial as spatial
from Bio.PDB import *
from tqdm import tqdm
from collections import Counter
from biopandas.pdb import PandasPdb
import linecache
import ctypes as c

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
    parser = ap.ArgumentParser(description="Encoding algorithm.")
    parser.add_argument("docking_program", type=str,
                        help="Path to folder containing the PDB files.")
    parser.add_argument("-c","--n_proc", type=int,
                        help='Number of processor.', default = 1)


    parsed_args = parser.parse_args()
    return parsed_args

def distances(ndarray_0, ndarray_1):
    if (ndarray_0.ndim, ndarray_1.ndim) not in ((1, 2), (2, 1)):
        raise ValueError("bad ndarray dimensions combination")
    return np.linalg.norm(ndarray_0 - ndarray_1, axis=1)

def get_most_dist_points(data, K, MAX_LOOPS=20):
    N = data.shape[0]
    ND = data.shape[1]
    indices = np.argsort(distances(data, data.mean(0)))[:K].copy()
    distsums = spatial.distance.cdist(data, data[indices]).sum(1)
    distsums[indices] = -np.inf
    prev_sum = 0.0
    for loop in range(MAX_LOOPS):
        for i in range(K):
            old_index = indices[i]
            distsums[old_index] = \
                distances(data[indices], data[old_index]).sum()
            distsums -= distances(data, data[old_index])
            new_index = np.argmax(distsums)
            indices[i] = new_index
            distsums[new_index] = -np.inf
            distsums += distances(data, data[new_index])
        curr_sum = spatial.distance.pdist(data[indices]).sum()
        if curr_sum == prev_sum:
            break
        prev_sum = curr_sum
    return indices


def get_ref_points(docking_program):
    file_names = [f'{docking_program}/' + f for f in os.listdir(f'{docking_program}/') if f.endswith(".pdb")]  
    ex_ptn = PandasPdb().read_pdb(file_names[0])
    df = ex_ptn.df['ATOM']
    df[df.chain_id == 'B'].head()
    dfB = df[df.chain_id == 'B'].set_index('atom_number')[['x_coord', 'y_coord', 'z_coord']]
    result = get_most_dist_points(dfB.values, K=3)
    i,j,k = dfB.iloc[result].index
    return [i,j,k]

def encode_file(file_name, atom_lines):
    l = count = 0
    df = pd.DataFrame(columns=('x', 'y', 'z'))
    for q,l in enumerate(atom_lines):
        line = linecache.getline(file_name[1], l+1)            
        df.loc[q] = [line[30:38], line[38:46], line[46:54]]
    array[file_name[0],:] = df.values.flatten()
    linecache.clearcache()

def main(args):
    """

    Parameters
    ----------
    args : argparse.Namespace
        It contains the command-line arguments that are supplied by the user    
    """
    global array
    # Fist step: Get reference for points
    i,j,k = get_ref_points(args.docking_program)
    
    # Second step: Ecoding
    file_names = [f'{args.docking_program}/' + f 
                  for f in os.listdir(f'{args.docking_program}/') 
                  if f.endswith(".pdb")]

    array = Array('d', np.zeros((len(file_names) *  9)), lock = False)
    
    def init_arr(array):
        globals()['array'] = \
            np.frombuffer(array, dtype='float').reshape(len(file_names), 9)
    
    # Paralelize the encoding loop
    encode_file_paral = partial(encode_file,
                                atom_lines = [i,j,k])
    
    Pool(args.n_proc, initializer=init_arr, initargs=(array,)).map(
        encode_file_paral, enumerate(file_names))
    
    result = np.frombuffer(array, dtype=float).reshape(len(file_names), 9)
    df_result = pd.DataFrame(result,
        columns=['x1','y1','z1','x2','y2','z2','x3','y3','z3'])
    df_result.to_csv(f'result_{args.docking_program}.csv', index=False)
     
if __name__ == '__main__':
    args = parse_args()
    main(args)