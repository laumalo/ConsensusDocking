"""
This module containts all the methods related with the aligment of 
PDB structures.
"""
from biopandas.pdb import PandasPdb
import numpy as np
import scipy.spatial as spatial
import torch 

class Aligner_3points(object): 
    """
    Aligner object based on the 3 CA method.
    """
    def __init__(self, pdb_ref, chain_ref): 
        """
        It initialices and Aligner object with the reference structure. 

        Parameters
        ----------
        pdb_ref : str
            Path to the reference PDB. 
        chain_ref : str     
            Chain Id of the reference structure. 
        """
        self.pdb_ref = pdb_ref
        self.chain_ref = chain_ref
        self.atoms_to_align_ref = \
            self.get_3points_lines(self.pdb_ref, self.chain_ref)


    def get_most_dist_points(self, data, K, MAX_LOOPS=20):
        """
        It gets the K most distance points of a given set of coordinates.

        Parameters
        ----------
        data : np.array
            Set of coordinates 
        K : int
            Number of points to return
        MAX_LOOPS : int
            Maximum number of loops in the algorithm. 

        Returns
        -------
        indices : np.array
            Indices of the K most distance points.
        """
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

    def get_3points_lines(self, pdb, chain):
        """
        For a given PDB and a chain ID it computes the three most distance CA. 

        Parameters
        ----------
        pdb : str
            Path to the PDB. 
        chain : str
            Chain ID. 

        Returns
        -------
        df : pandas.dataframe
            DataFrame with the 3 CA selected.
        """
        ppdb = PandasPdb().read_pdb(pdb)
        df = ppdb.df['ATOM'][ppdb.df['ATOM']['atom_name'] == 'CA']\
             [ppdb.df['ATOM']['chain_id'] == chain]
        coords = df[['x_coord', 'y_coord', 'z_coord']].values
        dist_atoms = self.get_most_dist_points(coords, K=3)
        return df.iloc[dist_atoms]


    def find_rigid_alignment(self, A, B):
        """
        It computes the rotation and translation matrix given to sets of 3D 
        points. At least each set has to containt 3 proints. Implementation 
        based on the Kabash algorithm. 

        Parameters
        ----------
        A : torch.tensor
            Set A of points. 
        B : torch.tensort
            Aet B of points.

        Returns
        -------
        R : np.array
            Rotation matrix. 
        t : np.array
            Translation vector.
        """
        A = torch.tensor(A)
        B = torch.tensor(B)
        
        a_mean = A.mean(axis=0)
        b_mean = B.mean(axis=0)
        A_c = A - a_mean
        B_c = B - b_mean
        
        # Covariance matrix
        H = A_c.T.mm(B_c)
        U, S, V = torch.svd(H)
        
        # Rotation matrix
        R = V.mm(U.T)
        
        # Translation vector
        t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
        t = t.T
        return np.array(R), np.array(t.squeeze())

    def align(self, pdb_query, chain_query): 
        """
        It performs a rigid aligment of the query chain against the reference 
        structure. This method is useful to align the static protein. 


        Parameters
        ----------
        pdb_query : str  
            Path to the PDB to align. 
        chain_query : str
            Chain ID to align. 
        """

        atoms_to_align_query = self.get_3points_lines(pdb_query, chain_query)

        A = self.atoms_to_align_ref[['x_coord', 'y_coord', 'z_coord']].values
        B = atoms_to_align_query[['x_coord', 'y_coord', 'z_coord']].values

        # Check if the structures are already aligned or not
        if (A==B).all(): 
            print(' -   Query and ref proteins already aligned.')
        else: 
            print(' -   Query and ref proteins are not aligned.')
            print(' -   Aligning query protein: {}...'.format(pdb_query))

            # Compute rotation and translation in 3D
            R,t = self.find_rigid_alignment(A, B)

            # Load PDB with original coordinates
            ppdb = PandasPdb().read_pdb(pdb_query)
            df = ppdb.df['ATOM']

            # Compute the new coordinates
            coords = df[['x_coord', 'y_coord', 'z_coord']].values
            new_coords = np.array([np.dot(R,coord)  + t for coord in coords])

            for i, (value, new_value) in enumerate(
                zip(df[['x_coord', 'y_coord', 'z_coord']].values, new_coords)): 
                df.at[i, 'x_coord'] = new_value[0]
                df.at[i, 'y_coord'] = new_value[1]
                df.at[i, 'z_coord'] = new_value[2]
            
            # Save aligned structure to PDB.
            ppdb.to_pdb(pdb_query.replace('.pdb', '_aligned.pdb'))

        def run_aligment(path, chain, n_proc = 1): 
            files = [os.path.join(path,file) for file in os.listdir(path) 
                     if file.endswith('.pdb')]
            align_paral = partial(align, chain = chain)
            with Pool(n_proc) as p:
                list(p.imap(align_paral, files))

class Aligner(object):
    """
    Aligner object based on MdTraj aligment of CA.
    """
    def __init__(self, pdb_ref, chain_ref): 
        """
        It initialices and Aligner object with the reference structure. 

        Parameters
        ----------
        pdb_ref : str
            Path to the reference PDB. 
        chain_ref : str     
            Chain Id of the reference structure. 
        """
        self.pdb_ref = pdb_ref
        self.chain_ref = chain_ref

    def align(query_structure, ref_traj):
        """
        It aligns a structure using MdTraj implementation. 

        Parameters
        ----------
        query_structure : str
            Path to the PDB of the structure to align.
        ref_traj : mdtraj.Trajectory object
            Reference trajectory. 
        """
        
        # Load reference receptor structure
        top_ref = ref_traj.topology
        atoms_to_align_ref = top_ref.select("chainid 0 and name CA")
        
        # Iterates over the docking poses generated for this swarm
        query_traj = md.load(query_structure)
        top_query = query_traj.topology
        atoms_to_align_query = top_query.select("chainid 0 and name CA")
        query_traj.superpose(ref_traj,
                             atom_indices = atoms_to_align_query,
                             ref_atom_indices = atoms_to_align_ref)
        output_path = query_structure.replace('.pdb', '_align.pdb')

        query_traj.save(output_path)
        os.remove(query_structure)

    def run_aligment(path, n_proc = 1):
        """
        It iterates (in parallel) over all the swam folders to aligned the 
        generated poses to the reference receptor structure. 
        """
        files = [os.path.join(path, file) for file in 
                 os.listdir(path) if file.endswith('.pdb')]
        ref_traj = md.load(self.pdb_ref)   
        align_structures_paral = partial(align_structures, ref_traj = ref_traj)
        with Pool(args.n_proc) as p:
            list(p.imap(align_structures_paral, files))