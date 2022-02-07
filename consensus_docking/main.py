
import os
import argparse as ap

import consensus_docking


def parse_args(args):
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
    parser = ap.ArgumentParser()
    parser.add_argument("conf_file", type=str,
                        help="Configuration file.")

    return parsed_args

def parse_conf_file(conf_file): 
    import configparser

    available_blocks = ['preprocessing', 'encoding', 'clustering.step1', 
                       'clustering.step2', 'analysis']
    config = configparser.ConfigParser()

    config.read('consensus.conf')
    consensus_blocks = config.sections()
        
    # Check conf file blocks
    checker = [block in available_blocks for block in consensus_blocks]
    if not all(checker): 
       print('Invalid configuration file')
    else:
        ordered_blocks = [block for block in available_blocks 
                          if block in consensus_blocks]
        print('You will run the following bocks:')
        for block in ordered_blocks:
            print(' - {}'.format(block))
    return ordered_blocks, config

def run_preprocessing(params): 
    pass 

def run_encoding(params): 
    pass 

def run_clustering(params):  
    pass 

def run_analysis(params): 
    pass 

def run_consensus_docking(args): 

	blocks, params = parse_conf_file(args.conf_file)
    
    for block in blocks:
        block_params = params[block]
        keys = [k for k in block_params]
        if block == 'preprocessing': 
            run_preprocessing(block_params)
        if block == 'encoding': 
            run_encoding(block_params)
        if block = 'clustering.step1' or block == 'clustering.step2': 
            run_clustering(block_params)
        if block == 'analysis': 
            run_analysis(block_params)


def main(args):
    """
    It reads the command-line arguments and runs the consensus docking protocol.
    Parameters
    ----------
    args : argparse.Namespace
        It contains the command-line arguments that are supplied by the user
    """
    run_consensus_docking(args)


if __name__ == '__main__':
    import sys
    args = parse_args(sys.argv[1:])
    main(args)