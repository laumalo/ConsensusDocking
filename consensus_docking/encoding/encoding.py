# General imports
from multiprocessing import Pool, Array
from functools import partial
import os
import numpy as np
import pandas as pd
import scipy.spatial as spatial
from biopandas.pdb import PandasPdb
import linecache


class Encoder(object):
    def __init__(self, docking_program, chain, working_dir='.'):
        self.docking_program = docking_program.lower()
        self.chain = chain
        self.working_dir = working_dir

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

        # N = data.shape[0]
        # ND = data.shape[1]
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
        pdb_path = os.path.join(self.working_dir, self.docking_program, pdb)
        ppdb = PandasPdb().read_pdb(pdb_path)
        df = ppdb.df['ATOM'][ppdb.df['ATOM']['atom_name'] == 'CA'][ppdb.df['ATOM']['chain_id'] == chain]
        coords = df[['x_coord', 'y_coord', 'z_coord']].values
        dist_atoms = self.get_most_dist_points(coords, K=3)
        return df.iloc[dist_atoms]

    def encode_file(self, file_name, atom_lines):
        try:
            # l = count = 0
            df = pd.DataFrame(columns=('x', 'y', 'z'))
            for q, l in enumerate(atom_lines):
                line = linecache.getline(file_name[1], l + 1)
                df.loc[q] = [line[30:38], line[38:46], line[46:54]]
            array[file_name[0], 2:] = df.values.flatten()
            linecache.clearcache()
            # print(array[file_name[0], :])
        except Exception as e:
            print(e)
            pass

    def run_encoding(self, output, n_proc=1):
        """

        """
        global array

        def init_arr(array):
            globals()['array'] = np.frombuffer(array, dtype='float').reshape(len(file_paths), 11)

        # Initialize array 
        file_paths = [f'{os.path.join(self.working_dir, self.docking_program, f)}'
                      for f in os.listdir(os.path.join(self.working_dir, self.docking_program))
                      if f.endswith(".pdb")]
        file_names = [f'{os.path.splitext(f)[0]}'
                      for f in os.listdir(os.path.join(self.working_dir, self.docking_program))
                      if f.endswith(".pdb")]

        array = Array('d', np.zeros((len(file_paths) * 11)), lock=False)

        # Get reference for points
        i, j, k = self.get_3points_lines(file_paths[0], self.chain)['line_idx'].values

        # Encoding
        encode_file_paral = partial(self.encode_file,
                                    atom_lines=[i, j, k])

        Pool(n_proc, initializer=init_arr, initargs=(array,)).map(
            encode_file_paral, enumerate(file_paths))

        # Write out results
        result = np.frombuffer(array, dtype=float).reshape(len(file_paths), 11)
        df_result = pd.DataFrame(result.astype(str),
                                 columns=['ids', 'norm_score', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3'])

        try:
            df_score = pd.read_csv(os.path.join(self.working_dir, self.docking_program, 'norm_score.csv'))
            score_ids = df_score.ids.to_list()
            for i, row in df_result.iterrows():
                encoding_id = file_names[i]
                df_result.at[i, 'ids'] = encoding_id
                if encoding_id in score_ids:
                    df_result.at[i, 'norm_score'] = float(df_score[df_score.ids == encoding_id].norm_score)
                else:
                    print(f'WARNING: No ids from norm_score coincided with file: {file_names[i]}. Setting 0 value')
            print('WARNING: If you haven\'t changed the file names of your files, norm_score column will be set to 0.')

        except FileNotFoundError:
            print(
                f'WARNING: norm_score.csv hasn\'t been found in {os.path.join(self.working_dir, self.docking_program)}'
                f', so energies won\'t be added to {output}.')
            for i, row in df_result.iterrows():
                encoding_id = file_names[i]
                df_result.at[i, 'ids'] = encoding_id
        df_result.to_csv(output, index=False)
