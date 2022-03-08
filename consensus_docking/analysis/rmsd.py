"""
It contains methods used to compute RMSD
"""
import os
import mdtraj as md
from functools import partial
from multiprocessing import Pool
from biopandas.pdb import PandasPdb
import numpy as np

from consensus_docking.utils.mdtraj import get_chains_dict

class RMSD(object): 
	def __init__(self):
		pass
	
	def _rmsd_ca_mdtraj(path_query, traj_ref, chain):
		"""

		"""
		
	    structure_query = md.load(path_query)
	    chain_id = get_chains_dict(path_query).get(chain)
	    top_query = \
	    	structure_query.topology.select(f"chainid {chain_id} and name CA")
	    traj_query = structure_query.atom_slice(top_query)
	    rmsd = md.rmsd(traj_ref, traj_query, precentered = True)[0] * 10
	    print(f'{os.path.basename(path_query)},{rmsd}')
	    return rmsd

	def _rmsd(path_query, ref, chain):
	    """RMSD CA with prealigned structutures"""
	    rep = PandasPdb().read_pdb(path_query)

	    ref_ca = ref.df['ATOM'].loc[(ref.df['ATOM'].atom_name == 'CA') \
	    		 & (ref.df['ATOM'].chain_id == chain)]
	    rep_ca = rep.df['ATOM'].loc[(rep.df['ATOM'].atom_name == 'CA') \
	    		 & (rep.df['ATOM'].chain_id == chain)]

	    assert ref_ca.shape[0] == rep_ca.shape[0], 'Different number of CA'
	    
	    add_squared_distaces = 0
	    for coord in ('x', 'y', 'z'):
	        ref_arr = np.array(ref_ca[f'{coord}_coord'])
	        rep_arr = np.array(rep_ca[f'{coord}_coord'])
	        add_squared_distaces += (ref_arr - rep_arr) ** 2
	    rmsd = \
	    	(np.sum(add_squared_distaces)/add_squared_distaces.shape[0]) ** 0.5
	    print(f'{os.path.basename(path_query)},{rmsd}')
	    return rmsd

	@staticmethod
	def compute_rmsd_ca_mdtraj(ref, path, chain = 'B' n_proc = 1):
	        files = [os.path.join(path,file) for file in os.listdir(path) 
	        		 if file.endswith('.pdb')]

	        # Load reference trajectory (only heavy atoms)
	        structure_ref = md.load(ref)
	        chain_id = get_chains_dict(ref).get(chain)
	        top_ref = \
	        	structure_ref.topology.select(f"chainid {chain_id} and name CA")
	        traj_ref = structure_ref.atom_slice(top_ref)
	        
	        compute_rmsd_paral = partial(rmsd_ca_mdtraj, traj_ref = traj_ref, 
	        							 chain = chain)
	        with Pool(n_proc) as p:
	                list(p.imap(compute_rmsd_paral, files))

	@staticmethod
	def compute_rmsd_ca(ref, path, chain ='B', n_proc = 1):
        files = [os.path.join(path,file) for file in os.listdir(path) 
        		 if file.endswith('.pdb')]

        # Load reference trajectory (only heavy atoms)
        structure_ref = PandasPdb().read_pdb(ref)
        compute_rmsd_paral = partial(compute_rmsd, ref=structure_ref,
        							 chain=chain)

        with Pool(n_proc) as p:
                list(p.imap(compute_rmsd_paral, files))
