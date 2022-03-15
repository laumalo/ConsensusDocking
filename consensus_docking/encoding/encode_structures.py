from consensus_docking.encoding import Encoder
import argparse as ap
import os


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
    parser = ap.ArgumentParser(description="Encoding algorithm.")
    parser.add_argument("docking_program", type=str,
                        help="Path to folder containing the PDB files.")
    parser.add_argument("output", type=str,
                        help="Path to the output file.")
    parser.add_argument("-c","--n_proc", type=int,
                        help='Number of processor.', default = 1)
    parser.add_argument("--chain", type=str,
                        help='Chain ID from the ligand protein.', default = 'B')
    parser.add_argument("--score", type=str, 
                        help='Path to normalized scoring file to add in the ' + 
                        'encoding.')

    parsed_args = parser.parse_args()
    return parsed_args


def main(args):
        encoder_tool = Encoder(docking_program = args.docking_program,
                                                   chain = args.chain)
        encoder_tool.run_encoding(output = args.output, score_file = args.score,
                                  n_proc = args.n_proc)

if __name__ == '__main__':
    args = parse_args()
    main(args)
