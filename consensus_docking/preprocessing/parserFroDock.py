import os
import pandas as pd
import numpy as np


class ParserFroDock:
    def __init__(self, working_dir='.', score_filename='dock.dat'):
        """
        It initializes a ParserFroDock object.
        Parameters
        ----------
        working_dir : str
            Path to the folder containing all the folders for each program.
        score_filename : str
            Name of the score file.
        """
        self.working_dir = working_dir
        self.program = 'frodock'
        self.score_filename = score_filename
        self.norm_score_filename = 'norm_score.csv'
        self.df = None

    @staticmethod
    def __find_starting_line(file_name):
        """
        It finds the line number in which the comment section ends, which indicates the beginning of the scoring section.
        Parameters
        ----------
        scoring_file_path : str
            Path to the scoring FroDock file

        Returns
        -------
        n : int
            Line number in which the comment section ends in the scoring file.
        """
        with open(file_name) as f:
            for n, line in enumerate(f):
                if n == 0 and line.startswith('More solutions requested'):
                    return [0]
                else:
                    return None

    def read(self):
        """
        It reads the scoring file and saves it to self.df
        """
        scoring_file_path = os.path.join(self.working_dir, self.program, self.score_filename)
        skip_rows = self.__find_starting_line(scoring_file_path)
        # skip_rows = [i for i in range(n+2)]
        header_list = ['rank_id', 'Euler1', 'Euler2', 'Euler3', 'posX', 'posY', 'PosZ', 'correlation']
        self.df = pd.read_csv(scoring_file_path, delimiter='\s+', skiprows=skip_rows, header=None, names=header_list)

    def __norm_scores(self):
        """
        It normalizes the score being 1 the best (the most negative) and 0 the worst (the most positive) and adds a new
        column to self.df with the normalized score
        """
        scores = np.array(self.df.correlation)
        self.df['norm_score'] = abs((scores - np.min(scores))) / (np.max(scores) - np.min(scores))

    def __norm_ids(self):
        """
        It normalizes the ids names (program_ + id number) and adds a new column to self.df with the normalized ids.
        """
        self.df['norm_ids'] = f'{self.program}_' + self.df.rank_id.map(str)

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
        columns_to_save = ['norm_ids', 'correlation', 'norm_score']
        header_names = ['ids', 'total_score', 'norm_score']
        if 'norm_score' not in self.df.columns:
            message = "You must normalize (sc_parser.norm()) before saving the csv with the normalized score."
            raise AttributeError(message)
        norm_score_file_path = os.path.join(self.working_dir, self.program, self.norm_score_filename)
        self.df.to_csv(norm_score_file_path, columns=columns_to_save, header=header_names, index=False)
