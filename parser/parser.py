"""
Parser module
"""


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
            from parserRosetta import ParserRosetta
            self.parser = ParserRosetta(self.working_dir, self.score_filename)
        elif self.program == 'piper':
            from parserPiper import ParserPiper
            self.parser = ParserPiper(self.working_dir, self.score_filename)
        elif self.program == 'frodock':
            from parserFroDock import ParserFroDock
            self.parser = ParserFroDock(self.working_dir, self.score_filename)
        elif self.program == 'patchdock':
            from parserPatchDock import ParserPatchDock
            self.parser = ParserPatchDock(self.working_dir, self.score_filename)
        elif self.program == 'lightdock':
            from parserLightDock import ParserLightDock
            self.parser = ParserLightDock(self.working_dir, self.score_filename)
        elif self.program == 'ftdock':
            from parserFTDock import ParserFTDock
            self.parser = ParserFTDock(self.working_dir, self.score_filename)
        elif self.program == 'zdock':
            from parserZDock import ParserZDock
            self.parser = ParserZDock(self.working_dir, self.score_filename)

    def run(self):
        """Parses and normalizes scoring files"""
        self.parser.read()
        self.parser.norm()
        self.parser.save()
