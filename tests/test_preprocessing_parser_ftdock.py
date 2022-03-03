import pytest
import pandas as pd
from consensus_docking.preprocessing import ParserFTDock
from .utils import *


@pytest.fixture
def parser():
    dir_name = os.path.dirname(__file__)
    base_dir = os.path.join(dir_name, 'data')
    return ParserFTDock(working_dir=base_dir, score_filename='dock.ene')


@pytest.fixture
def ref_norm_score_df():
    program = 'ftdock'
    relative_path = f'data/{program}/verified_{program}_norm_score.csv'
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             relative_path)
    return pd.read_csv(file_path)


class TestParserFTDock:
    """
    It wraps all tests that involve the ParserFTDock class
    """
    def test_initialization(self, tmp_path):
        """
        Test it can initialize the parser
        """
        sc_filename = 'dock.sc'
        program = 'ftdock'
        base_dir, *c = make_directory_structure(tmp_path, program, sc_filename)
        p = ParserFTDock(working_dir=base_dir, score_filename=sc_filename)
        assert isinstance(p, ParserFTDock)

    def test_validate_read(self, parser):
        """
        Test whether the parser can properly read a correct input file
        """
        parser.read()
        df_col_names = list(parser.df.columns)
        desired_col_names = ['Conf', 'Ele', 'Desolv', 'VDW', 'Total', 'RANK']
        for col in desired_col_names:
            assert col in df_col_names

    def test_invalid_sc_file(self, tmp_path):
        """
        Test that an error is raised when introducing an invalid score file
        """
        with pytest.raises(Exception):
            ParserFTDock(working_dir=tmp_path, score_filename='dummy')

    def test_norm(self, parser, ref_norm_score_df):
        """
        Test that the output file corresponds to the output file
        """
        parser.read()
        parser.norm()
        parser.df.reset_index(inplace=True)
        assert round(ref_norm_score_df['norm_score'], 10).\
            equals(round(parser.df['norm_score'], 10))

    def test_can_save_after_norm(self, parser, ref_norm_score_df, tmp_path):
        """
        Test that creates a non-empty output file and validates the output
        """
        parser.read()
        parser.norm()
        parser.save(tmp_path)
        norm_score_file_path = tmp_path / f'{parser.norm_score_filename}'
        out_df = pd.read_csv(norm_score_file_path)
        assert os.path.getsize(norm_score_file_path) > 0
        assert ref_norm_score_df.equals(out_df)

    def test_cannot_save_before_norm(self, parser, tmp_path):
        """
        Tests that an error is raised if the df is not normalized before saving
        the norm_scores.csv
        """
        parser.read()
        with pytest.raises(Exception):
            parser.save(tmp_path)
