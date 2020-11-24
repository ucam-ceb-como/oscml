import collections
import logging

import numpy as np
import pandas as pd
import sklearn
from tqdm import tqdm

import oscml.data.dataset_cep
import oscml.data.dataset_hopv15
import oscml.features.weisfeilerlehman
import oscml.models.model_gnn
from oscml.utils.util import smiles2mol

def path_cepdb_valid_smiles(root='.'):
    return root + '/data/processed/CEPDB_valid_SMILES.csv'

def path_cepdb_25000(root='.'):
    return root + '/data/processed/CEPDB_25000.csv'

def path_hopv_15(root='.'):
    return root + '/data/raw/HOPV_15_revised_2.data'

def path_osaka(root='.'):
    return root + '/data/raw/Nagasawa_RF_SI.txt'

class DataTransformer():
    
    def __init__(self, column_target, target_mean, target_std, column_x=None):
        self.column_target = column_target
        self.target_mean = target_mean
        self.target_std = target_std
        self.column_x = column_x

    def transform_x(self, data):
        if self.column_x:
            return data[self.column_x]
        else:
            return data

    def transform(self, data):
        if self.column_x:
            return (data[self.column_target] - self.target_mean) / self.target_std
        else:
            return (data - self.target_mean) / self.target_std

    def inverse_transform(self, data):
        if self.column_target:
            return data[self.column_target] * self.target_std + self.target_mean
        else:
            # that means isinstance(data, torch.Tensor) because the value predicted by PyTorch
            # has to be transformed back for evaluation
            return data * self.target_std + self.target_mean
    
def create_transformer(df, column_target, column_x=None):
    mean = float(df[column_target].mean())
    std = float(df[column_target].std(ddof=0))
    logging.info('calculated target mean=%s, target std=%s', mean, std)
    return DataTransformer(column_target, mean, std, column_x)

def add_node2index(original, new, zero_index_for_new):
             
    added = original.copy()
    for node in new:
        if node not in original:
            index = (0 if zero_index_for_new else len(added))
            added[node] = index
    return added

def clean_data(df, mol2seq, column_smiles, column_target):

    mask_known = []
    for i in tqdm(range(len(df))):
        smiles = df.iloc[i][column_smiles]
        m = smiles2mol(smiles)
        contains_only_known_types = True
        if mol2seq:
            try:
                mol2seq(m)
            except:
                contains_only_known_types = False
        mask_known.append(contains_only_known_types)

    mask_known = np.array(mask_known)
    logging.info('molecules with known atom types=%s', len(df[mask_known]))
    mask_notna = df[column_target].notna().to_numpy()
    logging.info('molecules with given target value for %s = %s', column_target, len(df[mask_notna]))
    mask = np.logical_and(mask_known, mask_notna)
    df_cleaned = df[mask].copy()
    logging.info('molecules with both=%s', len(df_cleaned))
    
    return df_cleaned

def store(df, filepath):
    logging.info('storing %s', filepath)
    # store without the internal index of Pandas Dataframe
    df.to_csv(filepath, index=False)

def split_data_frames_and_transform(df, column_smiles, column_target, train_size, test_size, seed):
    
    train_plus_val_size = len(df) - test_size
    df_train, df_test = sklearn.model_selection.train_test_split(df, 
                    train_size=train_plus_val_size, shuffle=True, random_state=seed)
    df_train, df_val = sklearn.model_selection.train_test_split(df_train, 
                    train_size=train_size, shuffle=True, random_state=seed+1)
    logging.info('train=%s, val=%s, test=%s', len(df_train), len(df_val), len(df_test))

    transformer = create_transformer(df_train, column_target, column_smiles)

    return df_train, df_val, df_test, transformer

def read_and_split(filepath, split_column='ml_phase'):
    logging.info('reading %s', filepath)
    df = pd.read_csv(filepath)
    df_train = df[(df[split_column] == 'train')].copy()
    df_val = df[(df[split_column] == 'val')].copy()
    df_test = df[(df[split_column] == 'test')].copy()
    logging.info('split data into sets of size (train / val / test)=%s / %s / %s', len(df_train), len(df_val), len(df_test))
    return df_train, df_val, df_test

def get_dataframes(dataset, type_dict, train_size=-1, test_size=-1, seed=200):

    src = dataset['src']
    x_column = dataset['x_column'][0]
    y_column = dataset['y_column'][0]

    if type_dict == oscml.data.dataset_hopv15.HOPV15:
        df = oscml.data.dataset_hopv15.read(src)
        df = oscml.data.dataset.clean_data(df, None, x_column, y_column)

        df_train, df_val, df_test, transformer = oscml.data.dataset.split_data_frames_and_transform(
                df, column_smiles=x_column, column_target=y_column, train_size=train_size, test_size=test_size, seed=seed), 
    
        return (df_train, df_val, df_test, transformer)

    elif type_dict == oscml.data.dataset_cep.CEP25000:
        df_train, df_val, df_test = oscml.data.dataset.read_and_split(src)
        # for testing only
        #df_train, df_val, df_test = df_train[:1500], df_val[:500], df_test[:500]
        transformer = oscml.data.dataset.create_transformer(df_train,
                column_target=y_column, column_x=x_column)

        return (df_train, df_val, df_test, transformer)
    
    raise RuntimeError('unknown dataset type dict=' + str(type_dict))

class DatasetInfo:
    def __init__(self, id=None, mol2seq=None, node_types=None, max_sequence_length=None, max_molecule_size=0, max_smiles_length=0):
        self.id=id
        #self.column_smiles = column_smiles
        #self.column_target = column_target
        if mol2seq:
            self.mol2seq = mol2seq
        else:    
            self.mol2seq = oscml.features.weisfeilerlehman.Mol2seq_WL(radius=1)
        if node_types:
            self.node_types = node_types
        else:
            self.node_types = collections.defaultdict(lambda:len(self.node_types))
        self.max_sequence_length = max_sequence_length
        self.max_molecule_size = max_molecule_size
        self.max_smiles_length = max_smiles_length
    
    def update(self, mol, smiles):
        self.mol2seq(mol)
        for a in mol.GetAtoms():
            node_type = (a.GetSymbol(), a.GetIsAromatic())
            self.node_types[node_type]
        self.max_molecule_size = max(self.max_molecule_size, len(mol.GetAtoms()))
        self.max_smiles_length = max(self.max_smiles_length, len(smiles))

    def number_subgraphs(self):
        return len(self.mol2seq.fragment_dict)

    def as_dict(self):
        d = {}
        d['id'] = self.id
        #d['column_smiles'] = self.column_smiles
        #d['column_target'] = self.column_target
        d['max_sequence_length'] = self.max_sequence_length
        d['max_molecule_size'] = self.max_molecule_size
        d['max_smiles_length'] = self.max_smiles_length
        d['node_types'] = dict(self.node_types)
        d['wf_r1'] = {
            'atom_dict': dict(self.mol2seq.atom_dict),
            'bond_dict': dict(self.mol2seq.bond_dict),
            'fragment_dict': dict(self.mol2seq.fragment_dict),
            'edge_dict': dict(self.mol2seq.edge_dict)
        }
        return d

def get_dataset_info(dataset):
    if dataset == oscml.data.dataset_cep.CEP25000:
        return oscml.data.dataset_cep.create_dataset_info_for_CEP25000()
    elif dataset == oscml.data.dataset_hopv15.HOPV15:
        return oscml.data.dataset_hopv15.create_dataset_info_for_HOPV15()
    
    raise RuntimeError('unknown dataset=' + str(dataset))
