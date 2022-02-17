"""
Parser module
"""
import os
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
            logging.debug("Successfully created ParserRosetta object")
        elif self.program == 'piper':
            from consensus_docking.preprocessing import ParserPiper
            self.parser = ParserPiper(self.working_dir, self.score_filename)
            logging.debug("Successfully created ParserPiper object")
        elif self.program == 'frodock':
            from consensus_docking.preprocessing import ParserFroDock
            self.parser = ParserFroDock(self.working_dir, self.score_filename)
            logging.debug("Successfully created ParserFroDock object")
        elif self.program == 'patchdock':
            from consensus_docking.preprocessing import ParserPatchDock
            self.parser = ParserPatchDock(self.working_dir, self.score_filename)
            logging.debug("Successfully created ParserPatchDock object")
        elif self.program == 'lightdock':
            from consensus_docking.preprocessing import ParserLightDock
            self.parser = ParserLightDock(self.working_dir, self.score_filename)
            logging.debug("Successfully created ParserLightDock object")
        elif self.program == 'ftdock':
            from consensus_docking.preprocessing import ParserFTDock
            self.parser = ParserFTDock(self.working_dir, self.score_filename)
            logging.debug("Successfully created ParserFTDock object")
        elif self.program == 'zdock':
            from consensus_docking.preprocessing import ParserZDock
            self.parser = ParserZDock(self.working_dir, self.score_filename)
            logging.debug("Successfully created ParserZDock object")
        else:
            available_programs = ['ftdock', 'zdock', 'lightdock', 'frodock', 'patchdock', 'piper', 'rosetta']
            logging.error(f"Program {self.program} is still no available."
                          f" Try with one of the followings: {available_programs}.")
            raise NameError(f"Program {self.program} is still no available.")

    def run(self, output_folder):
        """Parses and normalizes scoring files"""
        self.parser.read()
        logging.debug(f"File {os.path.join(self.working_dir, self.score_filename)} Successfully read!")
        self.parser.norm()
        logging.debug(f"File {os.path.join(self.working_dir, self.score_filename)} Successfully normalized!")
        self.parser.save(output_folder=output_folder)
        logging.info(f"Successfully created {self.parser.norm_score_filename} at the output folder!")
        logging.debug(f"Output path: {output_folder}")
