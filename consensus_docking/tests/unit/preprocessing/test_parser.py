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
