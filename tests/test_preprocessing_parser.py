import pytest
from consensus_docking.preprocessing import *
from .utils import *


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
        is_lightdock = False if program != 'lightdock' else True
        base_dir, *c = make_directory_structure(tmp_path, program, sc_filename,
                                                is_lightdock=is_lightdock)
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

