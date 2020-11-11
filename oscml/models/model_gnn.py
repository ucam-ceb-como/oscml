import collections
import datetime
import logging
from time import sleep

import dgl
import networkx 
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
import rdkit.Chem.rdmolops
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

import oscml.utils.params
from oscml.utils.params import cfg
import oscml.utils.util
from oscml.utils.util import log, concat
from oscml.utils.util import smiles2mol
import oscml.utils.util_pytorch
import oscml.utils.util_lightning

class Mol2seq_simple():
    
    def __init__(self, node2index={}, fix=False, oov=False):
        
        if fix:
            self.node2index = dict(node2index)
        else:
            self.node2index = collections.defaultdict(lambda:len(self.node2index), node2index)
            
        self.oov = oov
        # node index starts with 0, thus -1
        self.max_index = len(self.node2index) - 1
        logging.info(concat('initialized Mol2seq_simple with fix=', fix, ', oov=', oov, ', max_index=', self.max_index))
    
    def apply_OOV(self, index):
        return (index if index <= self.max_index else -1)
        
    def __call__(self, m):
        seq = []
        for a in m.GetAtoms():
            node = (a.GetSymbol(), a.GetIsAromatic())
            try:
                index = self.node2index[node]
                if self.oov:
                    index = self.apply_OOV(index)
            except KeyError as keyerror:
                if self.oov:
                    index = -1
                else:
                    raise keyerror
            seq.append(index)
        return seq

def create_dgl_graph(smiles, mol2seq_fct):
    m = smiles2mol(smiles)
    adj = rdkit.Chem.rdmolops.GetAdjacencyMatrix(m)
    g_nx = networkx.convert_matrix.from_numpy_matrix(adj)
    #g = dgl.DGLGraph(g_nx)
    g = dgl.from_networkx(g_nx)

    seq = mol2seq_fct(m)
    tensor = torch.as_tensor(seq, dtype=torch.long, device = cfg[oscml.utils.params.PYTORCH_DEVICE])
    g.ndata['type'] = tensor
    
    return g
    
class DatasetForGnnWithTransformer(torch.utils.data.Dataset):
    
    def __init__(self, df, mol2seq_fct, smiles_fct, target_fct):
        super().__init__()
        self.df = df
        self.mol2seq_fct = mol2seq_fct
        
        if isinstance(smiles_fct, str):
            self.smiles_fct = lambda data: data[smiles_fct]
        else:
            self.smiles_fct = smiles_fct
        if isinstance(target_fct, str):
            self.target_fct = lambda data: data[target_fct]
        else:
            self.target_fct = target_fct
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        smiles = self.smiles_fct(row)
        g = create_dgl_graph(smiles, self.mol2seq_fct)
        device = cfg[oscml.utils.params.PYTORCH_DEVICE]
        y = self.target_fct(row)
        y = torch.as_tensor(np.array(y, dtype=np.float32), device = device)
        return [g, y]
    
    def __len__(self):
        return len(self.df)
    
def collate_fn(data):
    graphs, y = map(list, zip(*data))
    g_batch = dgl.batch(graphs)
    device = cfg[oscml.utils.params.PYTORCH_DEVICE]
    y_batch = torch.as_tensor(y, device = device)
    return [g_batch, y_batch]

def get_dataloaders_internal(train, val, test, batch_size, mol2seq, transformer):
 
    smiles_fct = transformer.transform_x
    target_fct = transformer.transform

    train_dl = None
    if train is not None:
        train_ds = DatasetForGnnWithTransformer(train, mol2seq, smiles_fct, target_fct)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl = None
    if val is not None:
        val_ds = DatasetForGnnWithTransformer(val, mol2seq, smiles_fct, target_fct)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size, shuffle=False, collate_fn=collate_fn)
    test_dl = None
    if test is not None:
        test_ds = DatasetForGnnWithTransformer(test, mol2seq, smiles_fct, target_fct)
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False, collate_fn=collate_fn)
    
    batch_func = (lambda dl : len(dl) if dl else 0)
    batch_numbers = list(map(batch_func, [train_dl, val_dl, test_dl]))
    logging.info(concat('batch numbers - train val test=', batch_numbers))
 
    if test is None:
        return train_dl, val_dl 
    return train_dl, val_dl, test_dl

def get_dataloaders(dataset, df_train, df_val, df_test, transformer, batch_size):

    info = oscml.data.dataset.get_dataset_info(dataset)
    node2index = info.node_types
    mol2seq = oscml.models.model_gnn.Mol2seq_simple(node2index, fix=True, oov=True)
    return get_dataloaders_internal(df_train, df_val, df_test, batch_size, mol2seq, transformer)

    """
    if dataset == oscml.data.dataset_hopv15.HOPV15:

        info = oscml.data.dataset.get_dataset_info(dataset)
        node2index = info.node_types
        mol2seq = oscml.models.model_gnn.Mol2seq_simple(node2index, fix=True, oov=True)
        return get_dataloaders_internal(df_train, df_val, df_test, batch_size, mol2seq, transformer)

    elif dataset == oscml.data.dataset_cep.CEP25000:

        info = oscml.data.dataset.get_dataset_info(dataset)
        node2index = info.node_types
        mol2seq = oscml.models.model_gnn.Mol2seq_simple(node2index, fix=True, oov=True)
        return get_dataloaders_internal(df_train, df_val, df_test, batch_size, mol2seq, transformer)

    else:
        raise RuntimeError('unknown dataset=' + str(dataset))
    """

class GNNSimpleLayer(pl.LightningModule):
    
    def __init__(self, input_dim, output_dim, activation_fct):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation_fct = activation_fct
        self.msg = dgl.function.copy_src(src='h', out='m')
        self.reduce = dgl.function.sum(msg='m', out='h')
    
    def forward(self, g, h_input_features):
        with g.local_scope():
            g.ndata['h'] = h_input_features
            # TODO: check whether reduce includes the own current-h ('self loop')
            g.update_all(self.msg, self.reduce)
            h = g.ndata['h']
            h = self.linear(h)
            return self.activation_fct(h)

class GNNSimple(oscml.utils.util_lightning.OscmlModule):
    
    def __init__(self, node_type_number, conv_dim_list, mlp_dim_list, padding_index, target_mean, target_std, optimizer, optimizer_lr, mlp_dropout_list=None):

        super().__init__(optimizer, optimizer_lr, target_mean, target_std)
        logging.info('initializing ' + str(locals()))

        self.save_hyperparameters()

        if padding_index is not None:
            self.padding_index = padding_index
            logging.info(concat('padding index for embedding was set to ', self.padding_index, 
                '. Thus unknown atom types can be handled'))
            # consider the padding index for subsequential transfer learning with unknown atom types:
            # we add +1 to node_type_number because
            # padding_idx = 0 in a sequences is mapped to zero vector
            self.embedding = nn.Embedding(node_type_number+1, conv_dim_list[0], padding_idx=self.padding_index)
        else:
            self.padding_index = None
            logging.info('No padding index for embedding was set. No transfer learning for unknown atom types will be possible')
            self.embedding = nn.Embedding(node_type_number, conv_dim_list[0])
        
        self.conv_modules = nn.ModuleList()
        for i in range(len(conv_dim_list)-1):
            layer = GNNSimpleLayer(conv_dim_list[i], conv_dim_list[i+1], F.relu)
            self.conv_modules.append(layer)
            
        self.mlp = oscml.utils.util_pytorch.create_mlp(mlp_dim_list, mlp_dropout_list)
        
        self.one = torch.Tensor([1]).long().to(cfg['PYTORCH_DEVICE'])
    
    def forward(self, graphs):
        if isinstance(graphs, list):
            g_batch = dgl.batch(graphs)
        else:
            g_batch = graphs
        
        seq_types = g_batch.ndata['type']
        if self.padding_index is not None:
            minus_one =self.one.expand(seq_types.size())
            seq_types = seq_types + minus_one
            
        h = self.embedding(seq_types)       
        for layer in self.conv_modules:
            h = layer(g_batch, h)       
        g_batch.ndata['h'] = h
        
        mean = dgl.mean_nodes(g_batch, 'h')       
        o = self.mlp(mean)
        # transform from size [batch_number, 1] to [batch_number]
        o = o.view(len(o))   
        return o