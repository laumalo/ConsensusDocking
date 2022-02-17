import argparse as ap
from consensus_docking.preprocessing import Parser
import logging
import sys

logging.basicConfig(format='%(asctime)s [%(module)s] - %(levelname)s: %(message)s', datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO, stream=sys.stdout)


def parse_args():
    """
    It parses the command-line arguments.
    Parameters
    ----------
    args : list[str]
        List of command-line arguments to parse
    Returns
    -------
    parsed_args : argparse.Namespace
        It contains the command-line arguments that are supplied by the user
    """
    parser = ap.ArgumentParser(description="Parses scoring files of a program and normalizes them in a csv file.")
    parser.add_argument("program", type=str,
                        help="Protein-Protein Docking Program to be analyzed. If you want to analyze different "
                             "programs, make a coma-separated list")
    parser.add_argument("score_filename", type=str,
                        help="Name of the score file. If you decided to analyze different programs, make a "
                             "coma-separated list with the score filenames (in the same order!)")
    parser.add_argument("-w", "--w-dir", type=str,
                        help='Path to folder containing the folders of each program.', default='.')
    parser.add_argument("-o", "--output-folder", type=str,
                        help='Path to the output folder.', default='.')
    parsed_args = parser.parse_args()
    return parsed_args


def parse_sc(working_dir, program, score_filename, output_folder):
    """
    It parses the scoring file and generates a normalized file containing the pose id, the program score and the
    normalized score.
    Parameters
    ----------
    working_dir
    program
    score_filename
    output_folder
    """
    parser = Parser(program=program, score_filename=score_filename, working_dir=working_dir)
    parser.run(output_folder)


def main(args):
    """
    Calls parse_sc using command line arguments
    """
    available_programs = ['ftdock', 'zdock', 'lightdock', 'frodock', 'patchdock', 'piper', 'rosetta']
    input_programs = args.program.split(',')
    input_sc_filenames = args.score_filename.split(',')
    assert len(input_programs) == len(input_sc_filenames), "Program and Score filename arguments must have the same " \
                                                           "number of items!"
    for i, program in enumerate(input_programs):
        if program.lower() in available_programs:
            parse_sc(args.w_dir, program, input_sc_filenames[i], args.output_folder)
        else:

            logging.warning(f"Program {program} is still not available."
                            f" Try with one of the following programs: {available_programs}.")


if __name__ == '__main__':
    args = parse_args()
    main(args)
