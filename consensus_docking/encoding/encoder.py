from multiprocessing import Pool, Array
from functools import partial
import os
import numpy as np
import pandas as pd
import scipy.spatial as spatial
from biopandas.pdb import PandasPdb
import linecache
import logging
import sys

logging.basicConfig(
    format='%(asctime)s [%(module)s] - %(levelname)s: %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.INFO, stream=sys.stdout)


class Encoder:
    """ Encoder object """

    def __init__(self, docking_program, chain, docking_path=os.getcwd()):
        """
        It initializes an Encoder object.

        Parameters
        ----------
        docking_program : str
            Docking program name.
        chain : str
            Chain ID of the ligand protein. 
        docking_path : str
            Path to the docking folder. Default: working directory. 
        """
        self.path = docking_path
        self.docking_program = docking_program.lower()
        self.chain = chain

    @staticmethod
    def get_most_dist_points(data, K, MAX_LOOPS=20):
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

        indices = np.argsort(distances(data, data.mean(0)))[:K].copy()
        dist_sums = spatial.distance.cdist(data, data[indices]).sum(1)
        dist_sums[indices] = -np.inf
        prev_sum = 0.0
        for loop in range(MAX_LOOPS):
            for i in range(K):
                old_index = indices[i]
                dist_sums[old_index] = \
                    distances(data[indices], data[old_index]).sum()
                dist_sums -= distances(data, data[old_index])
                new_index = np.argmax(dist_sums)
                indices[i] = new_index
                dist_sums[new_index] = -np.inf
                dist_sums += distances(data, data[new_index])
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
        pdb_path = os.path.join(self.path, self.docking_program, pdb)
        ppdb = PandasPdb().read_pdb(pdb_path)
        df = ppdb.df['ATOM'][ppdb.df['ATOM']['atom_name'] == 'CA'] \
            [ppdb.df['ATOM']['chain_id'] == chain]
        coords = df[['x_coord', 'y_coord', 'z_coord']].values
        dist_atoms = self.get_most_dist_points(coords, K=3)
        return df.iloc[dist_atoms]

    @staticmethod
    def encode_file(file_name, atom_lines):
        """
        It encodes a file given the lines of the three most distant CA. 

        Parameters
        ---------
        file_name : str
            Path to the file to encode. 
        atom_lines : list
            Lines of the three most distant CA.
        """
        try:
            df = pd.DataFrame(columns=('x', 'y', 'z'))
            for q, l in enumerate(atom_lines):
                line = linecache.getline(file_name[1], l + 1)
                df.loc[q] = [line[30:38], line[38:46], line[46:54]]
            array[file_name[0], 2:] = df.values.flatten()
            linecache.clearcache()
        except Exception:
            logging.warning('Skipping file {}.'.format(file_name))

    def run_encoding(self, output, score_file=None, n_proc=1):
        """
        It runs the encoding of all the conformations found in the output
        docking folder. 

        Parameters
        ----------
        output : str
            Path to the output CSV file to save the encoding.
        score_file : str
            Path to the file containing the normalized scores.
        n_proc : int
            Number of processors.
        """
        global array

        def init_arr(array):
            globals()['array'] = \
                np.frombuffer(array, dtype='float').reshape(len(file_paths), 11)

        # Initialize array 
        file_paths = \
            [f'{os.path.join(self.path, self.docking_program, f)}'
             for f in os.listdir(os.path.join(self.path, self.docking_program))
             if f.endswith(".pdb")]
        file_names = \
            [f'{os.path.splitext(f)[0]}'
             for f in os.listdir(os.path.join(self.path, self.docking_program))
             if f.endswith(".pdb")]

        array = Array('d', np.zeros((len(file_paths) * 11)), lock=False)

        # Get reference for points
        i, j, k = \
            self.get_3points_lines(file_paths[0], self.chain)['line_idx'].values

        # Encoding
        encode_file_paral = partial(self.encode_file, atom_lines=[i, j, k])

        Pool(n_proc, initializer=init_arr, initargs=(array,)).map(
            encode_file_paral, enumerate(file_paths))

        # Save all the encoded coordinates into a dataframe
        encoding = \
            np.frombuffer(array, dtype=float).reshape(len(file_paths), 11)
        df_encoding = pd.DataFrame(encoding.astype(str),
                                   columns=['ids', 'norm_score', 'x1', 'y1',
                                            'z1', 'x2', 'y2', 'z2', 'x3', 'y3',
                                            'z3'])

        # Parse names and scores for each file
        if score_file is None or not os.path.exists(score_file):
            if score_file is None:
                logging.warning(f'     Norm_score path was NOT specified,' +
                                f' so energies won\'t be added to {output}')
            elif not os.path.exists(score_file):
                logging.warning(f'{score_file} was NOT FOUND.')

            for i, row in df_encoding.iterrows():
                encoding_id = file_names[i]
                df_encoding.at[i, 'ids'] = encoding_id

        else:
            df_score = pd.read_csv(score_file)
            score_ids = df_score.ids.to_list()
            for i, row in df_encoding.iterrows():
                encoding_id = file_names[i]
                df_encoding.at[i, 'ids'] = encoding_id
                if encoding_id in score_ids:
                    df_encoding.at[i, 'norm_score'] = \
                        float(df_score[df_score.ids == encoding_id].norm_score)
                else:
                    logging.warning(f'No ids from norm_score coincided with ' +
                                    f'file: {file_names[i]}. Setting 0 value.')

        df_encoding_sorted = df_encoding.sort_values('norm_score',
                                                     ascending=False)
        # Export output file
        df_encoding_sorted.to_csv(output, index=False)
