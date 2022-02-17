"""
Parser module
"""
import sys
import logging

logging.basicConfig(format='%(asctime)s [%(module)s] - %(levelname)s: %(message)s', datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO, stream=sys.stdout)


class Parser:
    def __init__(self, program, score_filename, working_dir='.'):
        """
        It initializes a specific Parser object for each program.
        Parameters
        ----------
        program : str
            Name of the program that wants to be parsed.
        score_filename : str
            Name of the score file.
        working_dir : str
            Path to the folder containing all the folders for each program.
        """
        self.program = program.lower()
        self.score_filename = score_filename
        self.working_dir = working_dir

        if self.program == 'rosetta':
            from consensus_docking.preprocessing import ParserRosetta
            self.parser = ParserRosetta(self.working_dir, self.score_filename)
        elif self.program == 'piper':
            from consensus_docking.preprocessing import ParserPiper
            self.parser = ParserPiper(self.working_dir, self.score_filename)
        elif self.program == 'frodock':
            from consensus_docking.preprocessing import ParserFroDock
            self.parser = ParserFroDock(self.working_dir, self.score_filename)
        elif self.program == 'patchdock':
            from consensus_docking.preprocessing import ParserPatchDock
            self.parser = ParserPatchDock(self.working_dir, self.score_filename)
        elif self.program == 'lightdock':
            from consensus_docking.preprocessing import ParserLightDock
            self.parser = ParserLightDock(self.working_dir, self.score_filename)
        elif self.program == 'ftdock':
            from consensus_docking.preprocessing import ParserFTDock
            self.parser = ParserFTDock(self.working_dir, self.score_filename)
        elif self.program == 'zdock':
            from consensus_docking.preprocessing import ParserZDock
            self.parser = ParserZDock(self.working_dir, self.score_filename)

    def run(self, output_folder):
        """Parses and normalizes scoring files"""
        self.parser.read()
        self.parser.norm()
        self.parser.save(output_folder=output_folder)
