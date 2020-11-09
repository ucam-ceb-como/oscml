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