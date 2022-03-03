import os
import sys
import logging
import pandas as pd
import numpy as np

logging.basicConfig(
    format='%(asctime)s [%(module)s] - %(levelname)s: %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.INFO, stream=sys.stdout)


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
        self._working_dir = None
        self.working_dir = working_dir
        self._program = 'frodock'
        self._score_filename = None
        self.score_filename = score_filename
        self._norm_score_filename = f'{self.program}_norm_score.csv'
        self._df = None

    @property
    def working_dir(self):
        self._working_dir

    @working_dir.setter
    def working_dir(self, new_working_dir):
        if isinstance(new_working_dir, str) and os.path.isdir(new_working_dir):
            self._working_dir = new_working_dir
        else:
            logging.error(f"Please enter a valid working_dir that exists. "
                          f"Keeping {self.working_dir}")

    @working_dir.getter
    def working_dir(self):
        return self._working_dir

    @property
    def program(self):
        self._program

    @program.getter
    def program(self):
        return self._program

    @property
    def score_filename(self):
        self._score_filename

    @score_filename.setter
    def score_filename(self, new_score_filename):
        folder_path = os.path.join(self.working_dir, self.program)
        file_path = os.path.join(folder_path, new_score_filename)
        if isinstance(new_score_filename, str) and os.path.exists(file_path):
            self._score_filename = new_score_filename
        else:
            logging.error(f"Please enter a valid score_filename that exists"
                          f" in {folder_path}. Keeping {self.score_filename}")

    @score_filename.getter
    def score_filename(self):
        return self._score_filename

    @property
    def norm_score_filename(self):
        self._norm_score_filename

    @norm_score_filename.getter
    def norm_score_filename(self):
        return self._norm_score_filename

    @property
    def df(self):
        self._df

    @df.setter
    def df(self, new_df):
        if isinstance(new_df, pd.DataFrame):
            self._df = new_df
        else:
            message = "Please enter a valid pd.DataFrame object. Keeping " \
                      "previous."
            logging.error(message)
            raise TypeError(message)

    @df.getter
    def df(self):
        return self._df

    @df.deleter
    def df(self):
        logging.warning("Removing df.")
        del self._df

    @staticmethod
    def __find_starting_line(scoring_file_path):
        """
        It finds the line number in which the comment section ends, which
        indicates the beginning of the scoring section.
        Parameters
        ----------
        scoring_file_path : str
            Path to the scoring FroDock file

        Returns
        -------
        n : int
            Line number in which the comment section ends in the scoring file.
        """
        with open(scoring_file_path) as f:
            for n, line in enumerate(f):
                if n == 0 and line.startswith('More solutions requested'):
                    return [0]
                else:
                    return None

    def read(self):
        """
        It reads the scoring file and saves it to self.df
        """
        scoring_file_path = os.path.join(self.working_dir, self.program,
                                         self.score_filename)
        skip_rows = self.__find_starting_line(scoring_file_path)
        # skip_rows = [i for i in range(n+2)]
        header_list = ['rank_id', 'Euler1', 'Euler2', 'Euler3', 'posX', 'posY',
                       'posZ', 'correlation']
        df = pd.read_csv(scoring_file_path, delim_whitespace=True, header=None,
                         skiprows=skip_rows)
        assert df.shape[1] == len(header_list), "Invalid FroDock scoring file," \
                                                f"expected {len(header_list)}" \
                                                f"columns, got {df.shape[1]}."
        df.set_axis(header_list, axis=1, inplace=True)
        self.df = df
        logging.debug(f"Scoring file read {scoring_file_path}: \n {self.df} ")

    def __norm_scores(self):
        """
        It normalizes the score being 1 the best (the most negative) and 0 the
        worst (the most positive) and adds a new column to self.df with the
        normalized score
        """
        scores = np.array(self.df.correlation)
        min_sc = np.min(scores)
        max_sc = np.max(scores)
        self.df['norm_score'] = abs((scores - min_sc)) / (max_sc - min_sc)
        logging.debug(f"Normalizing scores using: |scores - {min_sc} "
                      f"/ ( {max_sc} - {min_sc})|")

    def __norm_ids(self):
        """
        It normalizes the ids names (program_ + id number) and adds a new column
        to self.df with the normalized ids.
        """
        self.df['norm_ids'] = f'{self.program}_' + self.df.rank_id.map(str)

    def __sort_by_norm_score(self):
        """
        It sorts self.df by the normalized score value in descending order.
        """
        self.df = self.df.sort_values(by=['norm_score'], ascending=False)

    def norm(self):
        """
        It adds new columns to self.df normalizing scores and ids and finally
        sorts by norm_score in descending order.
        """
        self.__norm_scores()
        self.__norm_ids()
        self.__sort_by_norm_score()

    def save(self, output_folder):
        """
        It saves the normalized ids, the original score and the normalized score
        from self.df after being normalized to a csv file.
        """
        columns_to_save = ['norm_ids', 'correlation', 'norm_score']
        header_names = ['ids', 'total_score', 'norm_score']
        if 'norm_score' not in self.df.columns:
            message = "You must normalize (sc_parser.norm()) before saving " \
                      "the csv with the normalized score."
            raise AssertionError(message)
        norm_score_file_path = os.path.join(output_folder,
                                            self.norm_score_filename)
        self.df.to_csv(norm_score_file_path, columns=columns_to_save,
                       header=header_names, index=False)
