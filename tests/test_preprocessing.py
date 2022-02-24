import os
import pytest
import pandas as pd
from consensus_docking.preprocessing import *
from .utils import *


@pytest.fixture
def parser_ftdock():
    dir_name = os.path.dirname(__file__)
    base_dir = os.path.join(dir_name, 'data')
    return ParserFTDock(working_dir=base_dir, score_filename='dock.ene')


@pytest.fixture
def ref_norm_score_df_ftdock():
    relative_path = 'data/ftdock/verified_ftdock_norm_score.csv'
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             relative_path)
    return pd.read_csv(file_path)


class TestParser:
    """
    It wraps all tests that involve the Parser class
    """

    @pytest.mark.parametrize("program, parser_class",
                             [('zdock', ParserZDock),
                              ("piper", ParserPiper),
                              ("ftdock", ParserFTDock),
                              ("rosetta", ParserRosetta),
                              ("frodock", ParserFroDock),
                              ("lightdock", ParserLightDock),
                              ("patchdock", ParserPatchDock)])
    def test_can_set_correct_parser(self, program, parser_class, tmp_path):
        """
        Test that Parser makes the correct class instance according to program
        """
        sc_filename = 'dock.sc'
        base_dir, *c = make_directory_structure(tmp_path, program, sc_filename)
        p = Parser(program, sc_filename, working_dir=base_dir)
        assert p.parser.program == program
        assert isinstance(p.parser, parser_class)

    def test_cannot_initialize(self, tmp_path):
        """
        Test that Parser cannot be initialized because the working_dir or the
        score_file do not exist
        """
        sc_filename = 'dock.sc'
        program = 'ftdock'
        base_dir, *c = make_directory_structure(tmp_path, program, sc_filename)
        # test wrong score file
        with pytest.raises(Exception):
            Parser(program, 'dummy_sc_file', working_dir=base_dir)
        # test wrong working_dir
        with pytest.raises(Exception):
            Parser(program, sc_filename, working_dir='/dummy/path')

    def test_unexpected_program(self, tmp_path):
        """
        Tests that Parses raises an error when getting an unexpected program
        """
        sc_filename = 'dock.sc'
        program = 'dummy_program'
        base_dir, *c = make_directory_structure(tmp_path, program, sc_filename)
        with pytest.raises(Exception):
            Parser(program, sc_filename, working_dir=base_dir)

    def test_run_parser(self, tmp_path):
        """
        Verify that Parser returns a norm_score file that is not empty using
        parserFTDock
        """
        dir_name = os.path.dirname(__file__)
        base_dir = os.path.join(dir_name, 'data')
        p = Parser('ftdock', 'dock.ene', working_dir=base_dir)
        p.run(tmp_path)
        norm_score_file_path = tmp_path / f'{p.norm_score_filename}'
        assert os.path.getsize(norm_score_file_path) > 0


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
        base_dir, _, _ = make_directory_structure(tmp_path, program, sc_filename)
        p = ParserFTDock(working_dir=base_dir, score_filename=sc_filename)
        assert isinstance(p, ParserFTDock)

    def test_validate_read(self, parser_ftdock):
        """
        Test whether the parser can properly read a correct input file
        """
        parser_ftdock.read()
        df_col_names = list(parser_ftdock.df.columns)
        desired_col_names = ['Conf', 'Ele', 'Desolv', 'VDW', 'Total', 'RANK']
        for col in desired_col_names:
            assert col in df_col_names

    def test_invalid_sc_file(self, tmp_path):
        """
        Test that an error is raised when introducing an invalid score file
        """
        with pytest.raises(Exception):
            ParserFTDock(working_dir=tmp_path, score_filename='dummy')

    def test_norm(self, parser_ftdock, ref_norm_score_df_ftdock):
        """
        Test that the output file corresponds to the output file
        """
        parser_ftdock.read()
        parser_ftdock.norm()
        parser_ftdock.df.sort_index(inplace=True)
        assert round(ref_norm_score_df_ftdock['norm_score'], 10).\
            equals(round(parser_ftdock.df['norm_score'], 10))

    def test_can_save_after_norm(self, parser_ftdock, ref_norm_score_df_ftdock,
                                 tmp_path):
        """
        Test that creates a non-empty output file and validates the output
        """
        parser_ftdock.read()
        parser_ftdock.norm()
        parser_ftdock.save(tmp_path)
        norm_score_file_path = tmp_path / f'{parser_ftdock.norm_score_filename}'
        out_df = pd.read_csv(norm_score_file_path)
        assert os.path.getsize(norm_score_file_path) > 0
        assert ref_norm_score_df_ftdock.equals(out_df)

    def test_cannot_save_before_norm(self, parser_ftdock, tmp_path):
        """
        Tests that an error is raised if the df is not normalized before saving
        the norm_scores.csv
        """
        parser_ftdock.read()
        with pytest.raises(Exception):
            parser_ftdock.save(tmp_path)

