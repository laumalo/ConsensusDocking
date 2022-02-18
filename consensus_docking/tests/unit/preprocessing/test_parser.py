import os
import pytest
from consensus_docking.preprocessing import *


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
    def test_can_set_correct_parser(self, program, parser_class):
        """
        Test that Parser makes the correct class instance according to program
        """
        p = Parser(program, 'score_file.sc')
        assert p.parser.program == program
        assert isinstance(p.parser, parser_class)

    def test_unexpected_program(self):
        """
        Tests that Parses raises an error when getting an unexpected program
        """
        with pytest.raises(Exception):
            Parser('dummy_program', 'score_file.sc')

    def test_run_parser(self, tmp_path):
        """
        Verify that Parser returns a norm_score file that is not empty using
        parserFTDock
        """
        p = Parser('ftdock', 'dock.ene', working_dir='../../data')
        p.run(tmp_path)
        norm_score_file_path = tmp_path / f'{p.norm_score_filename}'
        assert os.path.getsize(norm_score_file_path) > 0


