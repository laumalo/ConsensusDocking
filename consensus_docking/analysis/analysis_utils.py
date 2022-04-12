import os
import yaml
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
import matplotlib.pyplot as plt
import seaborn as sns


def rmsd_CA(reference, rep):
    ref_ca = reference.df['ATOM'][reference.df['ATOM'].atom_name == 'CA']
    rep_ca = rep.df['ATOM'][rep.df['ATOM'].atom_name == 'CA']

    # All atoms rmsd
    # ref_ca = reference.df['ATOM']
    # rep_ca = rep.df['ATOM']
    assert ref_ca.shape[0] == rep_ca.shape[0], 'Different number of CA'
    add_squared_distaces = 0
    for coord in ('x', 'y', 'z'):
        ref_arr = np.array(ref_ca[f'{coord}_coord'])
        rep_arr = np.array(rep_ca[f'{coord}_coord'])
        add_squared_distaces += (ref_arr - rep_arr) ** 2
    return (np.sum(add_squared_distaces) / add_squared_distaces.shape[0]) ** 0.5


def get_dict_from_yaml(file_path):
    with open(file_path) as f:
        return yaml.full_load(f)


def parse_encoding():
    # from consensus_docking.encoding import Encoding
    pass

