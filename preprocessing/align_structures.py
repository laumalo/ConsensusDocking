import os 
import mdtraj as md
import argparse as ap
from multiprocessing import Pool

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
    parser = ap.ArgumentParser(description="Generate all PDBs.")
    parser.add_argument("path", type=str,
                        help="Path to the LightDock simulation simulation.")
    parser.add_argument("ref_receptor_pdb", type=str,
                        help="Path to reference receptor protein PDB to align the outputs.")
    parser.add_argument("-c","--n_proc", type=int,
                        help='Number of processor.', default = 1)

    parsed_args = parser.parse_args()
    return parsed_args

def align_structures(folder):
    """
    It aligns all the structures of a swarm output folder of a LightDock docking. 
    Take into account that it remove the oriignal (unaligned) structure to save space. 

    Parameters
    ----------
    folder : str
        Path to the swarm folder.
    """
    query_files = [os.path.join(folder[1],file) for file in 
                   os.listdir(folder[1]) if file.startswith('lightdock_')]
    
    # Load reference receptor structure
    ref_traj = md.load(args.ref_receptor_pdb)
    top_ref = ref_traj.topology
    atoms_to_align_ref = top_ref.select("chainid 0")
    
    # Iterates over the docking poses generated for this swarm
    for query_structure in query_files:
        query_traj = md.load(query_structure)
        top_query = query_traj.topology
        atoms_to_align_query = top_ref.select("chainid 0")
        query_traj.superpose(ref_traj, atom_indices = atoms_to_align_query,  
            ref_atom_indices = atoms_to_align_ref)

        output_path = query_structure.replace('.pdb', '_aligned.pdb')
        query_traj.save(output_path)
        os.remove(query_structure)

def main(args):
    """
    It iterates (in parallel) over all the swam folders to aligned the generated poses to the 
    reference receptor structure. 
    """
    folders = [os.path.join(args.path,folder) for folder in 
               os.listdir(args.path) if folder.startswith('swarm_')]

    with Pool(args.n_proc) as p:
        list(p.imap(align_structures, enumerate(folders)))

if __name__ == '__main__':
    args = parse_args()
    main(args)
