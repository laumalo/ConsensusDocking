import os
import sys
import logging
import pandas as pd
import numpy as np

logging.basicConfig(format='%(asctime)s [%(module)s] - %(levelname)s: %(message)s', datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO, stream=sys.stdout)


class ParserLightDock:
    def __init__(self, working_dir='.', score_filename='gso_100.out'):
        """
        It initializes a ParserLightDock object.
        Parameters
        ----------
        working_dir : str
            Path to the folder containing all the folders for each program.
        score_filename : str
            Name of the score file.
        """
        self.working_dir = working_dir
        self.program = 'lightdock'
        self.score_filename = score_filename
        self.norm_score_filename = 'norm_score.csv'
        self.df = None

    def __read_one_score_file(self, scoring_file_path):
        """
        Reads one score file from an swarm and save it to a dataframe
        Parameters
        ----------
        scoring_file_path : str
            Path to the scoring file

        Returns
        -------
        df : Pandas.DataFrame
            df containing the information in the scoring file and a column indicating to which swarm belongs the info.
        """
        header_list = ['swarm', 'RecID', 'LigID', 'Luciferin', 'Neighbor_number', 'Vision_Range', 'Scoring']
        df = pd.read_csv(scoring_file_path, skiprows=[0], header=None, delimiter=')', names=['c1', 'c2']).drop(
            columns='c1')
        df[header_list] = df['c2'].str.split(pat='\s+', expand=True)
        df = df.drop(columns=['c2'])
        swarm_start = scoring_file_path.find('swarm_')
        swarm_end = scoring_file_path.find(self.score_filename)
        df['swarm'] = scoring_file_path[swarm_start:swarm_end - 1]
        return df

    def read(self):
        """
        It reads the scoring file and saves it to self.df. Note that it iterates over all the swarm folders that appear
        in the `working_dir`/lightdock.
        """
        path_ligthdock = os.path.join(self.working_dir, self.program)
        scoring_files = [os.path.join(path_ligthdock, folder, self.score_filename) for folder in
                         os.listdir(path_ligthdock) if folder.startswith('swarm_')]
        dfsc_list = []
        for sc in scoring_files:
            dfsc = self.__read_one_score_file(sc)
            dfsc_list.append(dfsc)
        self.df = pd.concat(dfsc_list)

    def __norm_scores(self):
        """
        It normalizes the score being 1 the best (the most positive) and 0 the worst (the most negative) and adds a new
        column to self.df with the normalized score
        """
        scores = np.array(pd.to_numeric(self.df.Scoring))
        self.df['norm_score'] = abs((scores - np.min(scores))) / (np.max(scores) - np.min(scores))

    def __norm_ids(self):
        """
        It normalizes the ids names (program_ + id number) and adds a new column to self.df with the normalized ids.
        """
        self.df['norm_ids'] = f'{self.program}_' + self.df.swarm + '_' + self.df.Scoring.map(str)

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

    def save(self, output_path):
        """
        It saves the normalized ids, the original score and the normalized score from self.df after being normalized
        to a csv file.
        """
        columns_to_save = ['norm_ids', 'Scoring', 'norm_score']
        header_names = ['ids', 'total_score', 'norm_score']
        if 'norm_score' not in self.df.columns:
            message = "You must normalize (sc_parser.norm()) before saving the csv with the normalized score."
            raise AttributeError(message)
        norm_score_file_path = os.path.join(output_path, self.norm_score_filename)
        self.df.to_csv(norm_score_file_path, columns=columns_to_save, header=header_names, index=False)
