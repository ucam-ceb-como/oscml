import logging
from time import sleep

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

COLUMNS_HOPV15 = ['smiles', 'inchi', 
                  # experimental data:
                  'doi', 'inchikey', 'construction', 'architecture', 'complement', 'homo', 'lumo',
                  'electrochemicalgap', 'opticalgap', 'pce', 'voc', 'jsc', 'fillfactor',
                  'prunedsmiles', 
                  # conformer array
                  'numberconformers', 'conformers']

ATOM_TYPES_HOPV15 = {('C', False): 0,
             ('C', True): 1,
             ('S', True): 2,
             ('S', False): 3,
             ('O', False): 4,
             ('H', False): 5,
             ('N', False): 6,
             ('N', True): 7,
             ('Si', False): 8,
             ('F', False): 9,
             ('Se', True): 10,
             ('O', True): 11}

HOPV15 = 'HOPV15'

def read(filepath):
    logging.info('reading data from ' + filepath)
    
    molecules = []
    with open(filepath) as f:
        while True:
            molecule = [None] * len(COLUMNS_HOPV15)
            line = f.readline()
            if not line:
                break # end of file
            
            molecule[0] = line.strip() # smiles
            molecule[1] = f.readline().strip() # inchi
            split = f.readline().strip().split(',') # experimental data
            molecule[2:7] = split[:5]
            molecule[7:15] = map(float, split[5:])
            molecule[15] = f.readline().strip() # pruned smiles

            # iterate on conformers and read their data
            conformer_number = int(f.readline().strip())
            molecule[16] = conformer_number
            for _ in range(conformer_number):
                assert f.readline().startswith('Conformer')
                # TODO: move the following lines to another function
                for _ in range(int(f.readline().strip())):
                    # TODO: read atom data 
                    line = f.readline().strip()
                for _ in range(4):
                    # TODO: read DFT functional data
                    line = f.readline().strip()

            molecules.append(molecule)

    df = pd.DataFrame(molecules, columns = COLUMNS_HOPV15)
    logging.info('reading finished, number of molecules=' + str(len(df)))
    return df

def clean_hopv15(df):
    columns = ['smiles', 'homo','lumo','electrochemicalgap','opticalgap', 'pce']
    mask = df['smiles'].notna()
    for c in columns:
        mask = mask & df[c].notna()
    return df[mask]

def create_dataset_info_for_HOPV15():

    # the dictionary was created and logged during preprossing the entire CEPDB
    # it was copied manually here from the log file to fix the fragment-to-embedding-index mapping
    d = {'max_molecule_size': 53, 'max_smiles_length': 83, 'node_types': {('C', False): 0, ('C', True): 1, ('Se', True): 2, ('O', True): 3, ('N', True): 4, ('S', True): 5, ('H', False): 6, ('Si', False): 7}, 'wf_r1': {'atom_dict': {'C': 0, ('C', 'aromatic'): 1, ('Se', 'aromatic'): 2, ('O', 'aromatic'): 3, ('N', 'aromatic'): 4, ('S', 'aromatic'): 5, 'H': 6, 'Si': 7}, 'bond_dict': {'SINGLE': 0, 'DOUBLE': 1, 'AROMATIC': 2}, 'fragment_dict': {(0, ((0, 0), (0, 0), (6, 0), (6, 0))): 0, (0, ((0, 0), (0, 1), (6, 0))): 1, (0, ((0, 0), (0, 1), (1, 0))): 2, (1, ((0, 0), (1, 2), (4, 2))): 3, (1, ((1, 2), (1, 2), (6, 0))): 4, (1, ((1, 2), (1, 2), (2, 2))): 5, (2, ((1, 2), (1, 2))): 6, (1, ((1, 2), (1, 2), (3, 2))): 7, (3, ((1, 2), (1, 2))): 8, (1, ((1, 2), (3, 2), (6, 0))): 9, (1, ((1, 2), (1, 2), (1, 2))): 10, (1, ((1, 2), (1, 2), (4, 2))): 11, (4, ((1, 2), (5, 2))): 12, (5, ((4, 2), (4, 2))): 13, (1, ((1, 2), (4, 2), (6, 0))): 14, (4, ((1, 2), (1, 2))): 15, (6, ((0, 0),)): 16, (6, ((1, 0),)): 17, (1, ((0, 0), (1, 2), (1, 2))): 18, (1, ((0, 1), (1, 2), (1, 2))): 19, (0, ((1, 1), (6, 0), (7, 0))): 20, (7, ((0, 0), (0, 0), (6, 0), (6, 0))): 21, (6, ((7, 0),)): 22, (0, ((0, 0), (1, 1), (6, 0))): 23, (4, ((1, 2), (1, 2), (6, 0))): 24, (1, ((1, 2), (1, 2), (7, 0))): 25, (7, ((0, 0), (1, 0), (6, 0), (6, 0))): 26, (0, ((0, 0), (0, 1), (7, 0))): 27, (0, ((0, 1), (1, 0), (6, 0))): 28, (1, ((0, 0), (1, 2), (3, 2))): 29, (0, ((0, 0), (0, 0), (0, 1))): 30, (6, ((4, 0),)): 31, (0, ((0, 1), (6, 0), (7, 0))): 32, (1, ((0, 0), (1, 2), (2, 2))): 33, (1, ((1, 2), (2, 2), (6, 0))): 34, (1, ((1, 2), (5, 2), (6, 0))): 35, (5, ((1, 2), (1, 2))): 36, (1, ((1, 2), (1, 2), (5, 2))): 37, (1, ((0, 0), (1, 2), (5, 2))): 38, (0, ((0, 0), (1, 0), (6, 0), (6, 0))): 39, (0, ((0, 1), (1, 0), (7, 0))): 40, (1, ((4, 2), (5, 2), (6, 0))): 41, (1, ((0, 0), (4, 2), (4, 2))): 42, (1, ((4, 2), (4, 2), (6, 0))): 43, (1, ((0, 0), (4, 2), (5, 2))): 44, (1, ((1, 0), (1, 2), (4, 2))): 45, (1, ((1, 0), (1, 2), (3, 2))): 46, (1, ((1, 0), (1, 2), (5, 2))): 47, (1, ((1, 0), (1, 2), (1, 2))): 48, (1, ((1, 0), (1, 2), (2, 2))): 49, (1, ((1, 0), (4, 2), (4, 2))): 50, (1, ((1, 0), (4, 2), (5, 2))): 51, (0, ((4, 0), (6, 0), (6, 0), (6, 0))): 52, (4, ((0, 0), (1, 2), (1, 2))): 53, (0, ((1, 0), (1, 0), (6, 0), (6, 0))): 54, (7, ((1, 0), (1, 0), (6, 0), (6, 0))): 55}, 'edge_dict': {((0, 0), 0): 0, ((0, 6), 0): 1, ((0, 0), 1): 2, ((0, 1), 0): 3, ((1, 1), 2): 4, ((1, 4), 2): 5, ((1, 6), 0): 6, ((1, 2), 2): 7, ((1, 3), 2): 8, ((4, 5), 2): 9, ((0, 1), 1): 10, ((0, 7), 0): 11, ((6, 7), 0): 12, ((4, 6), 0): 13, ((1, 7), 0): 14, ((1, 5), 2): 15, ((1, 1), 0): 16, ((0, 4), 0): 17}}}
    mol2seq = Mol2seq_precalculated_with_OOV(radius=1, oov=True, wf=d['wf_r1'])
    logging.info('number of fragment types=' + str(len(mol2seq.fragment_dict)))         # 56
    logging.info('number_node_types=' + str(len(d['node_types'])))                      # 8

    params = {
        'id': HOPV15,
        'column_smiles': 'smiles',
        'column_target': 'pce',
        'mol2seq': mol2seq,
        'node_types': d['node_types'],
        'max_molecule_size': d['max_molecule_size'],                # 53
        'max_smiles_length': d['max_smiles_length'],                # 83
    }

    info = oscml.data.dataset.DatasetInfo(**params)
    return info