import os 
import argparse as ap
from multiprocessing import Pool
from functools import partial

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
    parser = ap.ArgumentParser(description="Aligment of PDB structures.")
    parser.add_argument("path", type=str,
                        help="Path to folder containing the PDBs to align.")
    parser.add_argument("chain_query", type=str,
                        help="Chain query.")
    parser.add_argument("ref_receptor_pdb", type=str,
                        help="Path to reference receptor protein PDB to align the outputs.")
    parser.add_argument("chain_ref", type=str,
                        help="Chain ref.")
    parser.add_argument("-c","--n_proc", type=int,
                        help='Number of processor.', default = 1)

    parsed_args = parser.parse_args()
    return parsed_args

def align_structure(file, aligner, chain):
    """
    It aligns the given PDB 
    Parameters
    ----------
    folder : str
        Path to the swarm folder.
    """
    aligner.align(pdb_query = file[1], chain_query = chain)

def main(args):
    """
    It iterates (in parallel) over all the PDB files in the folder to aligned the generated poses to the 
    reference structure. 
    """
    files = [os.path.join(args.path,file) for file in 
               os.listdir(args.path) if file.endswith('.pdb')]


    from align import Aligner 
    align_structures_paral = partial(align_structure,
                                aligner = Aligner(pdb_ref = args.ref_receptor_pdb, chain_ref = args.chain_ref),
                                chain = args.chain_query)
    with Pool(args.n_proc) as p:
        list(p.imap(align_structures_paral, enumerate(files)))

if __name__ == '__main__':
    args = parse_args()
    main(args)
