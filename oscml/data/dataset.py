import collections
import logging

import numpy as np
import pandas as pd
import sklearn
from time import sleep
from tqdm import tqdm

import oscml.models.model_gnn
from oscml.utils.util import concat
from oscml.utils.util import smiles2mol

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
    logging.info(concat('calculated target mean=', mean, ', target std=', std))
    return DataTransformer(column_target, mean, std, column_x)

def create_atom_dictionary(df, column_smiles, initial_dict = {}, with_aromaticity=True):
    
    # start with index 1 because index 0 is the padding index for embeddings
    d = collections.defaultdict(lambda:len(d) + 1, initial_dict)
    for i in tqdm(range(len(df))):
        smiles = df.iloc[i][column_smiles]
        m = smiles2mol(smiles)
        for a in m.GetAtoms():
            node_type = (a.GetSymbol(), a.GetIsAromatic())
            d[node_type]
    
    return d

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
    logging.info('molecules with known atom types=' + str(len(df[mask_known])))
    mask_notna = df[column_target].notna().to_numpy()
    logging.info(concat('molecules with given target value for ', column_target, '=', len(df[mask_notna])))
    mask = np.logical_and(mask_known, mask_notna)
    df_cleaned = df[mask].copy()
    logging.info('molecules with both=' + str(len(df_cleaned)))
    
    return df_cleaned

def get_dataloaders_with_calculated_normalized_data(df, column_smiles, column_target, args, train_size, test_size):
    
    mean = df[column_target].mean()
    std = df[column_target].std(ddof=0)
    logging.info(concat('target mean=', mean, 'target std=', std))
    transformer = DataTransformer(column_target, mean, std)
    
    x_train, x_test = sklearn.model_selection.train_test_split(df, 
                    train_size=(train_size + test_size), shuffle=True, random_state=0)
    x_train, x_val = sklearn.model_selection.train_test_split(x_train, 
                    train_size=train_size, shuffle=True, random_state=0)
    logging.info(concat('train=', len(x_train), ', val=', len(x_val), ', test=', len(x_test)))
    
    train_dl, val_dl, test_dl = oscml.models.model_gnn.get_dataloaders(x_train, x_val, x_test, args, 
                                                        column_smiles, transformer.transform)
    
    return train_dl, val_dl, test_dl, transformer.inverse_transform

def split_data_frames_and_transform(df, column_smiles, column_target, train_size, test_size):
    
    df_train, df_test = sklearn.model_selection.train_test_split(df, 
                    train_size=(train_size + test_size), shuffle=True, random_state=0)
    df_train, df_val = sklearn.model_selection.train_test_split(df_train, 
                    train_size=train_size, shuffle=True, random_state=0)
    logging.info(concat('train=', len(df_train), ', val=', len(df_val), ', test=', len(df_test)))

    transformer = create_transformer(df_train, column_target, column_smiles)

    return df_train, df_val, df_test, transformer

def read_and_split(filepath, split_column='ml_phase'):
    logging.info('reading ' +  filepath)
    df = pd.read_csv(filepath)
    df_train = df[(df[split_column] == 'train')].copy()
    df_val = df[(df[split_column] == 'val')].copy()
    df_test = df[(df[split_column] == 'test')].copy()
    logging.info(concat('split data into sets of size (train val test)=', len(df_train), len(df_val), len(df_test)))
    return df_train, df_val, df_test