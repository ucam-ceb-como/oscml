import collections
from time import sleep

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from oscml.utils.util import log
from oscml.utils.util import smiles2mol
import oscml.features.weisfeilerlehman
from oscml.features.weisfeilerlehman import mol2seq

ATOM_TYPES_CEP = {
     ('C', False): 0,
     ('C', True): 1,
     ('N', True): 2,
     ('S', True): 3,
     ('H', False): 4,
     ('Si', False): 5,
     ('O', True): 6,
     ('Se', True): 7
}

WL_R1_ATOM_DICT = {'Si': 0, 'C': 1, ('C', 'aromatic'): 2, ('S', 'aromatic'): 3, ('O', 'aromatic'): 4, 'H': 5, ('N', 'aromatic'): 6, ('Se', 'aromatic'): 7}

WL_R1_BOND_DICT = {'SINGLE': 0, 'DOUBLE': 1, 'AROMATIC': 2}

WL_R1_FRAGMENT_DICT = {(0, ((1, 0), (1, 0), (5, 0), (5, 0))): 0, (1, ((0, 0), (2, 1), (5, 0))): 1, (2, ((1, 1), (2, 2), (2, 2))): 2, (2, ((2, 0), (2, 2), (2, 2))): 3, (2, ((2, 2), (2, 2), (5, 0))): 4, (2, ((2, 2), (2, 2), (3, 2))): 5, (3, ((2, 2), (2, 2))): 6, (2, ((2, 2), (2, 2), (2, 2))): 7, (2, ((2, 2), (3, 2), (5, 0))): 8, (2, ((2, 0), (2, 2), (4, 2))): 9, (2, ((2, 2), (4, 2), (5, 0))): 10, (4, ((2, 2), (2, 2))): 11, (5, ((0, 0),)): 12, (5, ((1, 0),)): 13, (5, ((2, 0),)): 14, (2, ((2, 2), (5, 0), (6, 2))): 15, (6, ((2, 2), (2, 2), (5, 0))): 16, (5, ((6, 0),)): 17, (1, ((0, 0), (1, 1), (5, 0))): 18, (1, ((1, 0), (1, 1), (5, 0))): 19, (1, ((0, 0), (1, 1), (2, 0))): 20, (2, ((1, 0), (2, 2), (2, 2))): 21, (2, ((2, 2), (2, 2), (7, 2))): 22, (7, ((2, 2), (2, 2))): 23, (2, ((2, 2), (5, 0), (7, 2))): 24, (6, ((2, 2), (2, 2))): 25, (2, ((2, 2), (2, 2), (6, 2))): 26, (6, ((2, 2), (3, 2))): 27, (3, ((6, 2), (6, 2))): 28, (1, ((2, 0), (2, 0), (5, 0), (5, 0))): 29, (2, ((2, 0), (2, 2), (6, 2))): 30, (1, ((1, 0), (1, 0), (5, 0), (5, 0))): 31, (1, ((1, 0), (2, 1), (5, 0))): 32, (2, ((2, 0), (2, 2), (3, 2))): 33, (0, ((2, 0), (2, 0), (5, 0), (5, 0))): 34, (2, ((0, 0), (2, 2), (2, 2))): 35, (2, ((2, 2), (2, 2), (4, 2))): 36, (0, ((1, 0), (2, 0), (5, 0), (5, 0))): 37, (1, ((1, 1), (2, 0), (5, 0))): 38, (1, ((1, 0), (1, 1), (2, 0))): 39, (2, ((1, 0), (2, 2), (3, 2))): 40, (1, ((1, 0), (2, 0), (5, 0), (5, 0))): 41, (2, ((2, 0), (2, 2), (7, 2))): 42, (2, ((2, 0), (6, 2), (6, 2))): 43, (2, ((5, 0), (6, 2), (6, 2))): 44, (2, ((3, 2), (5, 0), (6, 2))): 45, (2, ((1, 0), (2, 2), (6, 2))): 46, (2, ((1, 0), (2, 2), (7, 2))): 47, (2, ((1, 0), (6, 2), (6, 2))): 48, (2, ((1, 0), (3, 2), (6, 2))): 49, (2, ((2, 0), (3, 2), (6, 2))): 50, (1, ((0, 0), (1, 0), (1, 1))): 51, (2, ((1, 0), (2, 2), (4, 2))): 52, (1, ((1, 0), (1, 0), (1, 1))): 53, (1, ((5, 0), (5, 0), (5, 0), (6, 0))): 54, (6, ((1, 0), (2, 2), (2, 2))): 55}

WL_R1_EDGE_DICT = {((0, 1), 0): 0, ((0, 5), 0): 1, ((1, 2), 1): 2, ((1, 5), 0): 3, ((2, 2), 2): 4, ((2, 2), 0): 5, ((2, 5), 0): 6, ((2, 3), 2): 7, ((2, 4), 2): 8, ((2, 6), 2): 9, ((5, 6), 0): 10, ((1, 1), 1): 11, ((1, 1), 0): 12, ((1, 2), 0): 13, ((2, 7), 2): 14, ((3, 6), 2): 15, ((0, 2), 0): 16, ((1, 6), 0): 17}


class Mol2seq_precalculated_with_OOV():
    
    def __init__(self, radius, oov):
        self.atom_dict = collections.defaultdict(lambda:len(self.atom_dict), WL_R1_ATOM_DICT)
        self.bond_dict = collections.defaultdict(lambda:len(self.bond_dict), WL_R1_BOND_DICT)
        self.fragment_dict = collections.defaultdict(lambda:len(self.fragment_dict), WL_R1_FRAGMENT_DICT)
        self.edge_dict = collections.defaultdict(lambda: len(self.edge_dict), WL_R1_EDGE_DICT)
        self.radius = radius
        self.oov = oov
        # fragment index starts with 0, thus -1
        self.max_index = len(self.fragment_dict) - 1
        log('initialized Mol2seq_precalculated_with_OOV with max_index=', self.max_index)
        
    def apply_OOV(self, index):
        return (index if index <= self.max_index else -1)
        
    def __call__(self, m):
        atoms, i_jbond_dict = oscml.features.weisfeilerlehman.get_atoms_and_bonds(m, self.atom_dict, self.bond_dict)
        descriptor = oscml.features.weisfeilerlehman.extract_fragments(self.radius, atoms, i_jbond_dict, self.fragment_dict, self.edge_dict)
        atoms_BFS_order = oscml.features.weisfeilerlehman.get_atoms_BFS(m)
        if self.oov:
            descriptor_BFS = [self.apply_OOV(descriptor[i]) for i in atoms_BFS_order]
        else:
            descriptor_BFS = [descriptor[i] for i in atoms_BFS_order]
        return descriptor_BFS

def mol2seq_precalculated_with_OOV(df, radius, oov, column_smiles='UNKNOWN'):
    log('filling mol2seq according to Weisfeiler Lehman algorithm with radius=', radius)
    sleep(1)
    
    mol2seq = Mol2seq_precalculated_with_OOV(radius, oov)
    
    if df is not None:
        for i in tqdm(range(len(df))):
            smiles = df.iloc[i][column_smiles]
            m = smiles2mol(smiles)
            mol2seq(m)
    
    log('atom dict:', len(mol2seq.atom_dict), mol2seq.atom_dict)
    log('bond dict:', len(mol2seq.bond_dict), mol2seq.bond_dict)
    log('fragment dict:', len(mol2seq.fragment_dict), mol2seq.fragment_dict)
    log('edge dict:', len(mol2seq.edge_dict), mol2seq.edge_dict)
    
    return mol2seq

def skip_invalid_smiles(df, smiles_column_name):
    smiles_valid = []
    max_smiles_length = 0
    max_atoms = 0
    for i in tqdm(range(len(df))):
        smiles = df.iloc[i][smiles_column_name]
        m = smiles2mol(smiles)
        valid = bool(m)
        smiles_valid.append(valid)
        i += 1
        if valid:
            max_smiles_length = max(max_smiles_length, len(smiles))
            max_atoms = max(max_atoms, len(m.GetAtoms()))

    df['SMILES_valid'] = smiles_valid      
    number_invalid_smiles = (df['SMILES_valid'] == False).sum()
    sleep(1)
    log('number of invalid SMILES=', number_invalid_smiles)
    mask = df['SMILES_valid']
    df = df[mask]
    log('number of selected DB entries==', len(df))
    log('max length of valid SMILES=', max_smiles_length)
    log('max number of atoms in molecules with valid SMILES=', max_atoms)
    
    return df.copy(), max_smiles_length, max_atoms

def skip_all_small_pce_values(df, threshold):
    mask = (df['pce'] >= threshold) 
    size_total = len(df)
    df = df[mask]
    size_larger = len(df)
    log('number of pce values smaller threshold=', size_total - size_larger)
    log('number of selected DB entries=', size_larger)
    return df.copy()

def sample_down_small_pce_values(df, threshold, percentage):
    """skip each sample with PCE value < DOWN_THRESHOLD 
    with probability DOWN_SKIP_PERCENTAGE
    DEPRECATED
    """
    mask = (df['pce'] < threshold)
    st = len(df[mask])
    log('ST = number of pce values smaller threshold =', st)
    log('expected number to skip = ST * percentage =', st * percentage)
    binomial = np.random.binomial(1, percentage, len(df))
    df['skip'] = binomial
    mask = (df['pce'] >= threshold) | (1 - df['skip'])
    df = df[mask]
    df = df.drop(['skip'], axis=1)
    log('number of selected DB entries=', len(df))
    return df.copy()

def clean_data(df, skip_invalid_smiles = False, min_row = None, max_row = None, 
               threshold_skip = None, threshold_downsampling = None, threshold_percentage = None):
    df_cleaned = df[min_row:max_row].copy()
    max_smiles_length = None
    max_smiles_atoms = None
    if skip_invalid_smiles:
        df_cleaned, max_smiles_length, max_smiles_atoms = skip_invalid_smiles(df_cleaned, 'SMILES_str')
    if threshold_skip:
        df_cleaned = skip_all_small_pce_values(df_cleaned, threshold_skip)
    if threshold_downsampling and threshold_percentage:
        df_cleaned = sample_down_small_pce_values(df_cleaned, threshold_downsampling, threshold_percentage)
    return df_cleaned, max_smiles_length, max_smiles_atoms

def sample_without_replacement(df, number_samples):
    # including endpoint 11.2
    bins = np.arange(0.0, 11.3, step=0.2)
    number_bins = len(bins)-1
    log('sampling without replacement, number of bins=', number_bins)
    #log('bins=', bins)
    labels = np.arange(number_bins)
    column_bin = pd.cut(df['pce'], bins=bins, labels=labels)
    # the next line is only for visualizing a bin diagram, you may comment it out
    df['bin'] = column_bin
    df_sampled, _ = train_test_split(df, train_size=number_samples, shuffle=True, 
                                                  random_state=0, stratify=column_bin)
    log('number of selected DB entries=', len(df_sampled))
    return df_sampled

class DataTransformer():
    
    def __init__(self, target_mean, target_std):
        self.target_mean = target_mean
        self.target_std = target_std
        
    def transform(self, data):
        result = (data - self.target_mean) / self.target_std
        return result
    
    def inverse_transform(self, data):
        """If the pcez values are predicted for evaluation (in the test phase), 
        the inverse Z-score transformation has to be applied to get the predicted values 
        before calculating mean errors or correlation coefficients.
        """
        result = data * self.target_std + self.target_mean
        return result
    
def split_and_normalize(df, train_ratio, val_ratio, test_ratio):
    df_train_plus_val_plus_test = df.copy()
    # sklearn is able to split pandas dataframe into smaller dataframes
    df_train_plus_val, df_test = train_test_split(df, test_size=test_ratio, shuffle=True) #, random_state=0)
    df_train_plus_val = df_train_plus_val.copy()
    df_test = df_test.copy()
    val_ratio = val_ratio / (train_ratio + val_ratio) 
    df_train, df_val = train_test_split(df_train_plus_val, test_size=val_ratio, shuffle=True) #, random_state=0)
    df_train = df_train.copy()
    df_val = df_val.copy()
    log('split data into sets of size (train val test)=', len(df_train), len(df_val), len(df_test))
    
    # normalize 
    pce_mean = df_train_plus_val['pce'].mean()
    pce_std = df_train_plus_val['pce'].std(ddof=0)
    log('normalizing PCE values with pce_mean=', pce_mean, 'pce_std=', pce_std)
    transformer = DataTransformer(pce_mean, pce_std)
    transform = transformer.transform
    
    df_train_plus_val_plus_test['pcez'] = transform(df_train_plus_val_plus_test['pce'])
    df_train_plus_val['pcez'] = transform(df_train_plus_val['pce'])
    df_train['pcez'] = transform(df_train['pce'])
    df_val['pcez'] = transform(df_val['pce'])
    # we also add column 'pcez' to df_test because normalizing the PCE values back and forth 
    # in the same way for train, val and test set simplifies some of the code
    df_test['pcez'] = transform(df_test['pce'])
    
    return transformer, df_train, df_val, df_test, df_train_plus_val, df_train_plus_val_plus_test

def read(filepath, threshold, number_samples):
    log('reading data from', filepath)
    df_cleaned = pd.read_csv(filepath)
    df_cleaned = skip_all_small_pce_values(df_cleaned, threshold)
    df_cleaned = sample_without_replacement(df_cleaned, number_samples)
    log('reading finished, number of molecules =', len(df_cleaned))
    return df_cleaned

def preprocess_CEP(filepath, threshold, number_samples, train_ratio, val_ratio, test_ratio):
    log('preprocessing data for args=', locals())
    df_train_plus_val_plus_test = read(filepath, threshold, number_samples)
    return split_and_normalize(df_train_plus_val_plus_test, train_ratio, val_ratio, test_ratio)

def store_CEP_with_valid_SMILES(path_source, path_dest, numbersamples=-1):
    log('reading', path_source)
    df_source = pd.read_csv(path_source)
    if numbersamples > 0:
        df_source = df_source[:numbersamples]
    df_dest, max_smiles_length, max_atoms = skip_invalid_smiles(df_source, 'SMILES_str')
    log('storing', path_dest)
    df_dest.to_csv(path_dest)