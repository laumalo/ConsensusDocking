
import os
import argparse as ap
import consensus_docking
import configparser
import logging
import sys 

logging.basicConfig(format="%(message)s", level=logging.INFO,
                    stream=sys.stdout)

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
    parser.add_argument("path", type=str,
                        help="Path to dockings results folder.")
    parser.add_argument("conf_file", type=str,
                        help="Configuration file.")
    parser.add_argument("-c", "--n_proc", type=int,
                        help='Number of processors.', default=1)

    parsed_args = parser.parse_args(args)
    return parsed_args

def parse_conf_file(conf_file):
    """
    Parses the configuration file to obtain all the parameters for each block. 

    Parameters
    ----------
    conf_file : str
        Path to configuration file. 

    Returns
    -------
    ordered_blocks : list
        Ordered list of the blocks to perform.
    config : condifparser object
        Configuration parameters for all the blocks.
    """

    AVAILABLE_BLOCKS = ['preprocessing', 'encoding', 'clustering.step1', 
                       'clustering.step2', 'analysis']
    config = configparser.ConfigParser()
    config.read(conf_file)
    consensus_blocks = config.sections()
        
    # Check configuration file blocks
    if not all([block in AVAILABLE_BLOCKS for block in consensus_blocks]): 
       logging.error('Invalid configuration file')
    else:
        ordered_blocks = [block for block in AVAILABLE_BLOCKS 
                          if block in consensus_blocks]
        logging.info('You will run the following blocks:')
        for block in ordered_blocks:
            logging.info('     - {}'.format(block))
    return ordered_blocks, config

def run_preprocessing(params, path, output_path, n_proc):
    """
    It runs the preprocessing block. 

    Parameters
    ----------
    params : condifparser object
        Configuration parameters to run the encoding.
    path : str
        Path to the working folder. 
    n_proc : int
        Number of processors. 
    """
    from consensus_docking.preprocessing import Aligner, Parser


    AVAILABLE_KEYS = ['reference', 'align', 'parser', 'scoring_files']
    keys = [k for k in params]
    folders = os.listdir(path)

    if not all([k in AVAILABLE_KEYS for k in keys]): 
        logging.error('Invalid preprocessing block configuration.')
    else: 
        logging.info(' - Running preprocessing:')

        # Aligment
        if 'align' in keys:
            if not 'reference' in keys: 
                logging.error('To align the structures you need to indicate' +  
                              'a reference.')
            else:
                logging.info('     Aligment of structures to {}:'
                             .format(params['reference']))
                folders_to_align = \
                    [folder.strip() for folder in list(params['align']
                    .split(','))]
                folder_chain_to_align = \
                    [list(f.split('_')) for f in folders_to_align]
                if not all([f in folders for f in folders_to_align]):
                    logging.error('Wrong selection of folders to align.')
                else: 
                    for folder, chains in folders_chian_to_align:
                        chains_to_align = list(chains.split())
                        logging.info('         Aligment of {}:'.format(folder))
                        folder_path = os.path.join(path, folder)
                        
                        aligner = Aligner(params['reference'], chains_to_align)
                        aligner.run_aligment(folder_path, chains_to_align,
                                             n_proc)

        # Parse scoring files
        if 'parser' in keys:
            if not 'scoring_files' in keys:
                logging.error('To parse the scorings you need to indicate' +
                              'the files.')
            else:
                logging.info('     Parsing scoring files.')
                folders_to_parse = \
                    [folder.strip() for folder in list(params['parse']
                        .split(','))]
                folders_file_to_parse = \
                    zip(folders_to_parse, 
                        list(params['scoring_files'].split(',')))

                if not all([f in folders for f in folders_to_parse]):
                    logging.info('Wrong selection of folders to parse' + 
                                 'the scorings.')
                else: 
                    for folder, file in folders_file_to_parse:
                        # Parsing
                        parser = Parser(folder, file, path)
                        parser.run(output_folder = output_path)


def run_encoding(params, path, output_path, n_proc):
    """
    It runs the encoding block. 

    Parameters
    ----------
    params : condifparser object
        Configuration parameters to run the encoding.
    path : str
        Path to the working folder. 
    n_proc : int
        Number of processors. 
    """
    import pandas as pd
    from consensus_docking.encoding import Encoder


    AVAILABLE_KEYS = ['encode', 'merge']
    keys = [k for k in params]

    available_folders = os.listdir(path)

    if not all([k in AVAILABLE_KEYS for k in keys]): 
        logging.error('Invalid encoding block configuration.')
    else:  
        # Encoding structures
        if 'encode' in keys: 
        logging.info(' - Running encoding:')
            folders_to_encode = \
                [folder.strip() for folder in list(params['encode'].split(','))]
            folder_chain_to_encode = \
                [list(f.split('_')) for f in folders_to_encode]
            
            if not all([f in available_folders for f in folders_to_parse]):
                logging.error('Wrong selection of folders to encode.')
            else:
                for folder, chain in folder_chain_to_encode: 
                    logging.info('     Encoding {} to {}'.format(folder,
                        '{}/encoding_{}.csv'.format(output_path, folder)))
                    
                    encoder = Encoder(folder, chain, path)
                    encoder.run_encoding(
                      output = '{}/encoding_{}.csv'.format(output_path, folder),
                      n_proc=n_proc)

        # Merging encodings
        if 'merge' in keys: 
            logging.info(' - Merging encodings:')
            encodings_to_merge = \
                [folder.strip() for folder in list(params['merge'].split(','))]
            files_to_merge = \
                [os.path.join(output_path,'encoding_{}.csv'.format(f))
                 for f in encodings_to_merge]
            
            # Combine and export all the selected encodings
            merged_csv_output = os.path.join(output_path,'merged_encoding.csv')
            merged_csv = pd.concat([pd.read_csv(f) for f in files_to_merge])
            merged_csv.to_csv(merged_csv_output,
                              index=False, encoding='utf-8-sig')
            logging.info('     Encoding saved to {}'.format(merged_csv_output))

def run_clustering(params, path, output_path, n_pro):
    """
    """
    AVAILABLE_KEYS = ['encoding_file','clustering_algorithm', 'eps', 'metric', 
                      'num_iter','keep_clusters']
    keys = [k for k in params]

    from consensus_docking.encoding import Encoding
    encoding = Encoding.from_csv(params['encoding_file'])
    encoding.df.columns = \
        ['File', 'Score','x1','y1','z1','x2','y2','z2','x3','y3','z3']
    coords = encoding.df[['x1','y1','z1','x2','y2','z2','x3','y3','z3']]
    
    from consensus_docking.clustering import Clustering
    clusters = Clustering(data = coords,
                          clustering_algorithm = params['clustering_algorithm']),
                           


def run_analysis(params, path, output_path, n_pro): 
    pass 


def outputs_handler(path): 
    """
    It handles the output paths. 

    Parameters
    ----------
    path : str
        Dockings path.
    """

    output_path = os.path.join(path, 'output')
    os.makedirs(output_path, exist_ok = True)

    preprocessing_output = os.path.join(output_path, 'preprocessing')
    os.makedirs(preprocessing_output, exist_ok = True)

    encodings_output = os.path.join(output_path, 'encodings')
    os.makedirs(encodings_output, exist_ok = True)

    clustering_output = os.path.join(output_path, 'clustering')
    os.makedirs(encodings_output, exist_ok = True)

    analysis_output = os.path.join(output_path, 'analysis')
    os.makedirs(analysis_output, exist_ok = True)


    return preprocessing_output, encodings_output, \
           clustering_output, analysis_output

def main(args):
    """
    It reads the command-line arguments and runs the consensus docking protocol.
    
    Parameters
    ----------
    args : argparse.Namespace
        It contains the command-line arguments that are supplied by the user
    """
    
    # Check docking conformations
    AVAILABLE_PROGRAMS = ['ftdock', 'zdock', 'lightdock', 'frodock',
                          'patchdock', 'piper', 'rosetta']
    programs = os.listdir(args.path)
    checker = all([program in AVAILABLE_PROGRAMS for program in programs])
    if not checker:
        logging.error('Wrong docking program.')
    
    else:
        # Handle outputs
        preprocessing_output, encodings_output, clustering_output, \
            analysis_output = outputs_handler(args.path)

        # Parse configuration file
        blocks, params = parse_conf_file(args.conf_file)
        
        # Run the diferent blocks of the workflow
        for block in blocks:
            if block == 'preprocessing': 
                run_preprocessing(params[block], args.path,
                                  preprocessing_output, args.n_proc)
            if block == 'encoding': 
                run_encoding(params[block], args.path,
                             encodings_output, args.n_proc)
            if block = 'clustering.step1' or block == 'clustering.step2': 
                run_clustering(params[block], args.path,
                               clustering_output, args.n_proc)
            if block == 'analysis': 
                run_analysis(params[block], args.path,
                         analysis_output, args.n_proc)


if __name__ == '__main__':
    import sys
    args = parse_args(sys.argv[1:])
    main(args)