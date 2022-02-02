from encoding import Encoder
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
    parser.add_argument("program_chain", type=str,
                        help="Protein-Protein Docking Program underscore chain of the ligand protein (eg. ftdock_B). "
                             "If you want to analyze different programs, make a coma-separated list without"
                             " white-spaces.")
    parser.add_argument("-w", "--w-dir", type=str,
                        help='Path to folder containing the folders of each program.', default='.')
    parser.add_argument("-ns", "--not-score", default=False, action='store_true',
                        help="If it is used, program_norm_score.csv  won't be checked, so energies won't be set in the"
                             " encoding output.")
    parser.add_argument("-c", "--n-proc", type=int,
                        help='Number of processor.', default=1)

    parsed_args = parser.parse_args()
    return parsed_args


def main(args):
    available_programs = ['ftdock', 'zdock', 'lightdock', 'frodock', 'patchdock', 'piper', 'rosetta']
    output_folder = os.path.join(args.w_dir, 'output', 'encoding')
    if not os.path.exists(output_folder):
        print(f'Creating {output_folder}.')
        os.makedirs(output_folder)
    input_programs_chain = args.program_chain.split(',')
    for program_chain in input_programs_chain:
        program, chain = program_chain.split('_')
        program = program.lower()
        chain = chain.upper()
        if program.lower() in available_programs:
            output_file = os.path.join(output_folder, f'{program}_encoding.csv')
            if args.not_score is False:
                norm_score_file = os.path.join(args.w_dir, 'output', 'norm_scores', f'{program}_norm_score.csv')
            else:
                norm_score_file = None
            encoder_tool = Encoder(docking_program=program, chain=chain, working_dir=args.w_dir)
            encoder_tool.run_encoding(output=output_file, norm_score_file=norm_score_file, n_proc=args.n_proc)
            # TODO Add option to merge some encodings for the clustering
        else:
            print(f"Program {program} is still no available. Try with one of the followings: {available_programs}.")


if __name__ == '__main__':
    args = parse_args()
    main(args)
