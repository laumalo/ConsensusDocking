import os
import pandas as pd
import numpy as np

class ParserPiper:
    def __init__(self, working_dir='.', score_filename='ft.000.00'):
        """
        It initializes a ParserPiper object.
        Parameters
        ----------
        working_dir : str
            Path to the folder containing all the folders for each program.
        score_filename : str
            Name of the score file.
        """
        self.working_dir = working_dir
        self.program = 'piper'
        self.score_filename = score_filename
        self.norm_score_filename = 'norm_score.csv'
        self.df = None

    def read(self):
        """
        It reads the scoring file and saves it to self.df
        """
        scoring_file_path = os.path.join(self.working_dir, self.program, self.score_filename)
        header_list = ['rot_index', 'x_trans', 'y_trans', 'z_trans', 'total_score', 'repulsive_vdw', 'attractive_vdw',
                       'coulomb_elect', 'born_approx_elect', 'pairwise_potential']
        self.df = pd.read_csv(scoring_file_path, delimiter='\s+', header=None, names=header_list)

    def __norm_scores(self):
        """
        It normalizes the score being 1 the best (the most negative) and 0 the worst (the most positive) and adds a new
        column to self.df with the normalized score
        """
        scores = np.array(self.df.total_score)
        self.df['norm_score'] = abs((scores - np.max(scores))) / (np.max(scores) - np.min(scores))

    def __norm_ids(self):
        """
        It normalizes the ids names (program_ + id number) and adds a new column to self.df with the normalized ids.
        """
        self.df['norm_ids'] = f'{self.program}_' + self.df.rot_index.map(str)

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
        columns_to_save = ['norm_ids', 'total_score', 'norm_score']
        header_names = ['ids', 'total_score', 'norm_score']
        if 'norm_score' not in self.df.columns:
            message = "You must normalize (parser.norm()) before saving the csv with the normalized score."
            raise AttributeError(message)
        norm_score_file_path = os.path.join(self.working_dir, self.program, self.norm_score_filename)
        self.df.to_csv(norm_score_file_path, columns=columns_to_save, header=header_names, index=False)
