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

def get_patches_CA(folder):
    """
    Given a folder with the different patches, it parses the information of the 
    different patches into a dictionary. 

    Parameters
    ----------
    folder : str
        Path to the patches folder. 

    Returns
    -------
    d_residues : dict
        Dictionary with the residue information for each patch. 
    """
    patches_path = os.listdir(folder)
    d_residues = {}
    for patch in patches_path: 
        path = os.path.join(folder, patch)
        ppdb = PandasPdb().read_pdb(path)
        df = ppdb.df['ATOM']
        d_residues[patch] = df[['residue_name', 'residue_number']].values
    return d_residues

def get_coords_CA(ppdb, chain, residue_name, residue_number):
    """
    It returns the coordinates of the CA given its residue information.

    Parameters
    ----------
    ppdb : biopandas.pdb.pandas_pdb.PandasPdb object
        PDB structure. 
    chain : str
        Chain Id. 
    residue_name : str
        Residue name. 
    residue_number : 
        Residue number. 

    Returns
    -------
    coords : list
        CA coordinates. 
    """
    df =  ppdb.df['ATOM'][ppdb.df['ATOM']['chain_id'] == chain] \
          [ppdb.df['ATOM']['atom_name'] == 'CA'][ppdb.df['ATOM'] \
          ['residue_name'] == residue_name] \
          [ppdb.df['ATOM']['residue_number'] == residue_number]
    return df[['x_coord', 'y_coord', 'z_coord']].values
    
def distances(ndarray_0, ndarray_1):
    """
    It computes the distance between two arrays of coordinates. 

    Parameters
    ----------
    ndarray_0 : np.array
        Array 1
    ndarray_1 : np.array
        Array 2
    """
    if (ndarray_0.ndim, ndarray_1.ndim) not in ((1, 2), (2, 1)):
        raise ValueError("bad ndarray dimensions combination")
    return np.linalg.norm(ndarray_0 - ndarray_1, axis=1)

def filter_structure(file_to_parse, topology, 
                     d_proteinA, d_proteinB, cutoff):
    
    """
    Checks if the given structure is between the filtering criteria.

    Parameters
    ----------
    file_to_parse : str
        Path to the structure's file.
    topology : str
        Path to the topology file, if needed. 
    d_proteinA : dict
        Dictionary with patches for protein A. 
    d_proteinB : dict
        Dictionary with patches for protein B. 
    cutoff : int
        Distance threshold (in A). 

    Returns
    -------
    keep : bool
        True if we keep the structure.
    """
    if file_to_parse.endswith('.pdb'):
        ppdb = PandasPdb().read_pdb(file_to_parse)

    if file_to_parse.endswith('xtc'):
        m = md.load(file_to_parse, top = topology)
        
        with tempfile.NamedTemporaryFile(suffix='.pdb') as tmp:
            m.save(tmp.name)
            ppdb = PandasPdb().read_pdb(tmp.name)

    for patchA, patchB in list(itertools.product(d_proteinA, d_proteinB)): 
        residues_proteinA = d_proteinA[patchA]
        residues_proteinB = d_proteinB[patchB]
        l1 = [get_coords_CA(ppdb, 'A', residue[0], residue[1]) 
              for residue in residues_proteinA]
        l2 = [get_coords_CA(ppdb, 'C', residue[0], residue[1]) 
              for residue in residues_proteinB]
        
        value = min(np.linalg.norm(l1_element - l2_element) 
                    for l1_element,l2_element in itertools.product(l1,l2))
        if value < cutoff:
            print(os.path.basename(file_to_parse), value)



def main(args):
    """
    Filters using the patches predicted by MaSIF-site all the poses fetched. 
    """

    d_proteinA = get_patches_CA(args.patches_receptor)
    d_proteinB = get_patches_CA(args.patches_ligand)
    
    files_pdb = [os.path.join(args.path, file) for file in os.listdir(args.path)
                 if file.endswith('pdb')]

    files_xtc = [os.path.join(args.path, file) for file in os.listdir(args.path)
                 if file.endswith('xtc')]

    files = files_xtc if bool(files_xtc) else files_pdb
    topology = files_pdb[0] if bool(files_xtc) else None

    filter_structure_paral = partial(filter_structure, 
                                     topology = topology,
                                     d_proteinA = d_proteinA, 
                                     d_proteinB = d_proteinB, 
                                     cutoff = args.filter)

    with Pool(args.n_proc) as p:
        list(p.imap(filter_structure_paral, files))

if __name__ == '__main__':
    args = parse_args()
    main(args)
