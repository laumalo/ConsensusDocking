import os 
import itertools
import numpy as np
from biopandas.pdb import PandasPdb
import argparse as ap
from multiprocessing import Pool
from functools import partial
import mdtraj as md
import tempfile

import logging 
logging.getLogger().setLevel(logging.INFO)

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
    parser = ap.ArgumentParser(description="Filtering using MaSIF patches.")
    parser.add_argument("path", type=str,
                        help="Path to folder containing the PDBs or XTCs to " +  
                        "filter using MaSIF patches")
    parser.add_argument("patches_receptor", type=str,
                        help="Path to folder with the patches of the " + 
                        "receptor protein.")
    parser.add_argument("patches_ligand", type=str,
                        help="Path to folder with the patches of the " + 
                        "ligand protein.")
    parser.add_argument("-c","--n_proc", type=int,
                        help='Number of processor.', default = 1)
    parser.add_argument("-f","--filter", type=int,
                        help='Cutoff distance to filter in A.', default = 15)
    parsed_args = parser.parse_args()
    return parsed_args

def main(args):
    """
    Filters using the patches predicted by MaSIF-site all the poses fetched. 
    """
    from consensus_docking.filtering import FilterMASIF
    masif_filter = FilterMASIF(path = args.path,
                               filter_receptor = args.patches_receptor,
                               filter_ligand = args.patches_ligand)
    masif_filter.run_filtering(filter_distance = args.filter, 
                               n_proc = args.n_proc)

if __name__ == '__main__':
    args = parse_args()
    main(args)
