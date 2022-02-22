"""
Parser module
"""
import os
import sys
import logging

logging.basicConfig(
    format='%(asctime)s [%(module)s] - %(levelname)s: %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
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
        self._working_dir = None
        self.working_dir = working_dir
        self._program = program.lower()
        self._score_filename = None
        self.score_filename = score_filename

        if self.program == 'rosetta':
            from consensus_docking.preprocessing import ParserRosetta
            self._parser = ParserRosetta(self.working_dir, self.score_filename)
            logging.debug("Successfully created ParserRosetta object")
        elif self.program == 'piper':
            from consensus_docking.preprocessing import ParserPiper
            self._parser = ParserPiper(self.working_dir, self.score_filename)
            logging.debug("Successfully created ParserPiper object")
        elif self.program == 'frodock':
            from consensus_docking.preprocessing import ParserFroDock
            self._parser = ParserFroDock(self.working_dir, self.score_filename)
            logging.debug("Successfully created ParserFroDock object")
        elif self.program == 'patchdock':
            from consensus_docking.preprocessing import ParserPatchDock
            self._parser = ParserPatchDock(self.working_dir, self.score_filename)
            logging.debug("Successfully created ParserPatchDock object")
        elif self.program == 'lightdock':
            from consensus_docking.preprocessing import ParserLightDock
            self._parser = ParserLightDock(self.working_dir, self.score_filename)
            logging.debug("Successfully created ParserLightDock object")
        elif self.program == 'ftdock':
            from consensus_docking.preprocessing import ParserFTDock
            self._parser = ParserFTDock(self.working_dir, self.score_filename)
            logging.debug("Successfully created ParserFTDock object")
        elif self.program == 'zdock':
            from consensus_docking.preprocessing import ParserZDock
            self._parser = ParserZDock(self.working_dir, self.score_filename)
            logging.debug("Successfully created ParserZDock object")
        else:
            available_programs = ['ftdock', 'zdock', 'lightdock', 'frodock',
                                  'patchdock', 'piper', 'rosetta']
            logging.error(f"Program {self.program} is still no available."
                          f" Try with one of the followings: {available_programs}.")
            raise NameError(f"Program {self.program} is still no available.")
        self._norm_score_filename = self.parser.norm_score_filename

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
            logging.error(f"Please enter a valid score_filename that exists "
                          f"in {folder_path}. Keeping {self.score_filename}")

    @score_filename.getter
    def score_filename(self):
        return self._score_filename

    @property
    def parser(self):
        self._parser

    @parser.getter
    def parser(self):
        return self._parser

    @property
    def norm_score_filename(self):
        self._norm_score_filename

    @norm_score_filename.getter
    def norm_score_filename(self):
        return self._norm_score_filename

    def run(self, output_folder):
        """Parses and normalizes scoring files"""
        self.parser.read()
        file_path = os.path.join(self.working_dir, self.score_filename)
        logging.debug(f"File {file_path} Successfully read!")
        self.parser.norm()
        logging.debug(f"File {file_path} Successfully normalized!")
        self.parser.save(output_folder=output_folder)
        logging.info(f"Successfully created {self.parser.norm_score_filename} "
                     f"at the output folder!")
        logging.debug(f"Output path: {output_folder}")
