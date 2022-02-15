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


class Filter(objet): 
    def __init__(self, path, filter_receptor,  filter_ligand): 
        self.path = path
        self.filter_receptor = filter_receptor
        self.filter_ligand = filter_ligand


    def get_patches_CA(self, folder):
        """
        Given a folder with the different patches, it parses the information 
        of the different patches into a dictionary. 

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

    def get_coords_CA(self, ppdb, chain, residue_name, residue_number):
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
        

    def filter_structure(self, file_to_parse, topology, 
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
            l1 = [get_coords_CA(ppdb, 'A', residue[0], residue[1]) 
                  for residue in d_proteinA[patchA]]
            l2 = [get_coords_CA(ppdb, 'C', residue[0], residue[1]) 
                  for residue in d_proteinB[patchB]]
            
            value = min(np.linalg.norm(l1_element - l2_element) 
                        for l1_element,l2_element in itertools.product(l1,l2))
            if value < cutoff:
                print(os.path.basename(file_to_parse), value)



    def run(self, filter_distance = 15, n_proc = 1):
        """
        Filters using the patches predicted by MaSIF-site all the poses fetched. 
        """

        d_proteinA = get_patches_CA(self.filter_receptor)
        d_proteinB = get_patches_CA(self.filter_ligand)
        
        files_pdb = [os.path.join(self.path, file) 
                     for file in os.listdir(self.path) if file.endswith('pdb')]

        files_xtc = [os.path.join(self.path, file)
                     for file in os.listdir(self.path) if file.endswith('xtc')]

        files = files_xtc if bool(files_xtc) else files_pdb
        topology = files_pdb[0] if bool(files_xtc) else None

        filter_structure_paral = partial(self.filter_structure, 
                                         topology = topology,
                                         d_proteinA = d_proteinA, 
                                         d_proteinB = d_proteinB, 
                                         cutoff = filter_distance)

        with Pool(n_proc) as p:
            list(p.imap(filter_structure_paral, files))