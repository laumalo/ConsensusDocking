import os
import sys
import logging
import pandas as pd
import numpy as np

logging.basicConfig(format='%(asctime)s [%(module)s] - %(levelname)s: %(message)s', datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO, stream=sys.stdout)


class ParserPatchDock:
    def __init__(self, working_dir='.', score_filename='output.txt'):
        """
        It initializes a ParserPatchDock object.
        Parameters
        ----------
        working_dir : str
            Path to the folder containing all the folders for each program.
        score_filename : str
            Name of the score file.
        """
        self.working_dir = working_dir
        self.program = 'patchdock'
        self.score_filename = score_filename
        self.norm_score_filename = 'norm_score.csv'
        self.df = None

    @staticmethod
    def __find_starting_line(scoring_file_path):
        """
        It finds the line number in which the comment section ends, which indicates the beginning of the scoring section.
        Parameters
        ----------
        scoring_file_path : str
            Path to the scoring PatchDock file

        Returns
        -------
        n : int
            Line number in which the comment section ends in the scoring file.
        """
        with open(scoring_file_path) as f:
            for n, line in enumerate(f):
                if line.startswith('**********') and not n == 0:
                    return n

    def read(self):
        """
        It reads the scoring file and saves it to self.df
        """
        scoring_file_path = os.path.join(self.working_dir, self.program, self.score_filename)
        n = self.__find_starting_line(scoring_file_path)
        skip_rows = [i for i in range(n + 3)]
        header_list = ['conf_index', 'score', 'pen.', 'Area', 'as1', 'as2', 'as12', 'ACE', 'hydroph', 'Energy',
                       'cluster', 'dist.', 'empty', 'Ligand Transformation']
        self.df = pd.read_csv(scoring_file_path, delimiter='\s*\|', skiprows=skip_rows, header=None,
                              names=header_list).drop(columns='empty')

    def __norm_scores(self):
        """
        It normalizes the score being 1 the best (the highest value) and 0 the worst (the lowest value) and adds a new
        column to self.df with the normalized score
        """
        scores = np.array(self.df.score)
        self.df['norm_score'] = abs((scores - np.min(scores))) / (np.max(scores) - np.min(scores))

    def __norm_ids(self):
        """
        It normalizes the ids names (program_ + id number) and adds a new column to self.df with the normalized ids.
        """
        self.df['norm_ids'] = f'{self.program}_' + self.df.conf_index.map(str)

    def __sort_by_norm_score(self):
        """
        It sorts self.df by the normalized score value in descending order.
        """
        self.df = self.df.sort_values(by=['norm_score'], ascending=False)

    def norm(self):
        """
        It adds new columns to self.df normalizing scores and ids and finally sorts by norm_score in descending order.
        """
        self.__norm_scores()
        self.__norm_ids()
        self.__sort_by_norm_score()

    def save(self):
        """
        It saves the normalized ids, the original score and the normalized score from self.df after being normalized
        to a csv file.
        """
        columns_to_save = ['norm_ids', 'score', 'norm_score']
        header_names = ['ids', 'total_score', 'norm_score']
        if 'norm_score' not in self.df.columns:
            message = "You must normalize (sc_parser.norm()) before saving the csv with the normalized score."
            raise AttributeError(message)
        norm_score_file_path = os.path.join(self.working_dir, self.program, self.norm_score_filename)
        self.df.to_csv(norm_score_file_path, columns=columns_to_save, header=header_names, index=False)
