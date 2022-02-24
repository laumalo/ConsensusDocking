import os 
import itertools
import numpy as np
from biopandas.pdb import PandasPdb
from multiprocessing import Pool
from functools import partial
import mdtraj as md
import tempfile
import pandas as pd 
import sys

import logging 
logging.basicConfig(
    format='%(asctime)s [%(module)s] - %(levelname)s: %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',level=logging.INFO, stream=sys.stdout)



class _Filter(object): 
    """ 
    It is the Filter base class
    """
    _filtering_method = ''

    def __init__(self): 
        """
        It initialicesthe base Filter class. 

        Parameters
        ----------
        path : str
            Path to the folder containing the structures to filter.
        """
        
        self.filtered_structures = None

    def _get_coords_CA(self, ppdb, chain, residue_name, residue_number):
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
            Dictionary with filtering atoms sets for protein A. 
        d_proteinB : dict
            Dictionary with filtering atoms sets for protein B. 
        cutoff : int
            Distance threshold (in A). 

        """
        if file_to_parse.endswith('.pdb'):
            ppdb = PandasPdb().read_pdb(file_to_parse)

        if file_to_parse.endswith('xtc'):
            m = md.load(file_to_parse, top = topology)
            
            with tempfile.NamedTemporaryFile(suffix='.pdb') as tmp:
                m.save(tmp.name)
                ppdb = PandasPdb().read_pdb(tmp.name)

        for patchA, patchB in list(itertools.product(d_proteinA, d_proteinB)): 
            l1 = [self._get_coords_CA(ppdb, 'A', residue[0], residue[1]) 
                  for residue in d_proteinA[patchA]]
            l2 = [self._get_coords_CA(ppdb, 'C', residue[0], residue[1]) 
                  for residue in d_proteinB[patchB]]
            
            value = min(np.linalg.norm(l1_element - l2_element) 
                        for l1_element,l2_element in itertools.product(l1,l2))
            if value < cutoff:
                return os.path.basename(file_to_parse), value
                break


    def filter_encoding_file(self, encoding_file, file_filtered = None): 
        """
        Given an encoding file, it filteres the encoding based on a list of 
        structures and generated a new encoding file. 

        Parameters
        ----------
        encoding_file : str
            Path to the encoding file. 
        file_filtered : str
            Path to a file containing the list of filtered structures. 
        """
        if file_filtered:
            with open(file_filtered, 'r') as f: 
                lines = f.readlines()
                filtered_structures = [line.replace('\n', '') 
                                       for line in lines if not 'gpfs' in line]
        else: 
            filtered_structures = self.filtered_structures

        df_encoding = pd.read_csv(encoding_file)
        df_filtered = df[df['File'].isin(filtered_structures)]

        out_file = file_filtered.replace('.csv', '_filtered.csv')
        df_filtered.to_csv(out_file)

    def run(self, d_proteinA, d_proteinB, filter_distance, output , n_proc = 1):
        """
        Filters the structures fetched with the filtering criteria (list of CA)
        of the two proteins to be under a certain filtering distance.

        Parameters
        ----------
        d_proteinA : dict
            Dictionary with filtering atoms sets for protein A. 
        d_proteinB : dict
            Dictionary with filtering atoms sets for protein B. 
        filter_distance : float
            Filtering distance. 
        output : str
            Path to output file.
        n_proc : int
            Number of processors to run it in parallel. 
        """

        logging.info('  - Filtering structures based on {}'
            .format(self._filtering_method))
        
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
            self.filtered_structure = p.map(filter_structure_paral, files)

        logging.info('  -   {}/{} structures filtered'
            .format(len(self.filtered_structure), len(files)))
        
        with open(output, 'w') as f:
            for structure in self.filtered_structure:
                f.write("%s\n" % structure)
        logging.info('  -   Filtered structures information saved as {}'
            .format(output))


class FilterMASIF(_Filter): 
    """
    It defines a MaSIF filtering. 
    """
    _filtering_method = 'MASIF'
    
    def __init__(self, path, filter_receptor,  filter_ligand):
        """
        It initializes the FilterMASIF class.

        Parameters
        ----------
        path : str
            Path to the folder containing structures to be filtered. 
        filter_receptor : str
            Path to the patches for the receptor protein. 
        filter_ligand : str
            Path to the patches for the ligand protein.             
        """

        self.filter_receptor = filter_receptor
        self.filter_ligand = filter_ligand
        
        super().__init__(self._filtering_method)

    def _get_patches_CA(self, folder):
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

    def run_filtering(self, filter_distance = 15, 
                      output = 'filtering_masif.csv', n_proc = 1): 
        """
        Filters the structures fetched with the filtering criteria (list of CA)
        of the two proteins to be under a certain filtering distance.

        Parameters
        ----------
        filter_distance : float
            Filtering distance. Default: 15 A
        output : str
            Path to output file. Default: filtering_masif.csv
        n_proc : int
            Number of processors to run it in parallel. Default: 1 
        """
        
        d_patchA = self._get_patches_CA(self.filter_receptor)
        d_patchB = self._get_patches_CA(self.filter_ligand)
        
        self.run(d_proteinA = d_patchA, 
                 d_proteinB = d_patchB, 
                 filter_distance = filter_distance, 
                 output = output, 
                 n_proc = n_proc)