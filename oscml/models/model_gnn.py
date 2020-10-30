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
from oscml.utils.util import log
from oscml.utils.util import logm
from oscml.utils.util import smiles2mol
import oscml.utils.util_pytorch
import oscml.utils.util_lightning

def init_params(df, node2index=None):
        
    log('initializing parameters for PyTorch model GNNSIMPLE')
    d = {}
    cfg['GNNSIMPLE'] = d
    if node2index:
        mol2seq = Mol2seq_simple(node2index, fix=True, oov=True)
    else:
        mol2seq = mol2seq_simple(df)
    d['MOL2SEQ'] = mol2seq       
    #print('MY', list(mol2seq.node2index.values()))
    node_type_number = 1 + max(list(mol2seq.node2index.values()))
    #print('MY', node_type_number)
    d['NODE_TYPE_NUMBER'] = node_type_number
    #d['NODE_TYPE_NUMBER'] = len(mol2seq.node2index)
    d['CONV_DIM_LIST'] = [128, 128, 128, 128, 128]
    d['MLP_DIM_LIST'] = [128, 128, 64, 64, 1]
    d['BATCH_SIZE'] = 250
    d['LEARNING_RATE'] = 0.001
    d['PADDING_INDEX'] = 0
    log('parameters for PyTorch model GNNSIMPLE:', d)

class Mol2seq_simple():
    
    def __init__(self, node2index={}, fix=False, oov=False):
        
        if fix:
            self.node2index = dict(node2index)
        else:
            self.node2index = collections.defaultdict(lambda:len(self.node2index), node2index)
            
        self.oov = oov
        # node index starts with 0, thus -1
        self.max_index = len(self.node2index) - 1
        log('initialized Mol2seq_simple with fix=', fix, ', oov=', oov, ', max_index=', self.max_index)
    
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

def mol2seq_simple(df, column='SMILES_str'):
    log('filling mol2seq_simple')
    sleep(1)
    
    mol2seq = Mol2seq_simple() 
    for i in tqdm(range(len(df))):
        smiles = df.iloc[i][column]
        m = smiles2mol(smiles)
        mol2seq(m)
    
    log('node2index dict:', len(mol2seq.node2index), mol2seq.node2index)
    
    return mol2seq

def create_dgl_graph(smiles, mol2seq_fct):
    m = smiles2mol(smiles)
    adj = rdkit.Chem.rdmolops.GetAdjacencyMatrix(m)
    g_nx = networkx.convert_matrix.from_numpy_matrix(adj)
    g = dgl.DGLGraph(g_nx)
    
    seq = mol2seq_fct(m)
    tensor = torch.as_tensor(seq, dtype=torch.long, device = cfg[oscml.utils.params.PYTORCH_DEVICE])
    g.ndata['type'] = tensor
    
    return g
    
class DatasetPceForGNNsimple(torch.utils.data.Dataset):
    
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

def get_dataloaders(df_train, df_val, df_test, args, smiles_fct='SMILES_str', target_fct='pcez'):

    mol2seq = args['MOL2SEQ']
    batch_size = args['BATCH_SIZE']
    
    train_dl = None
    if df_train is not None:
        train_ds = DatasetPceForGNNsimple(df_train, mol2seq, smiles_fct, target_fct)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl = None
    if df_val is not None:
        val_ds = DatasetPceForGNNsimple(df_val, mol2seq, smiles_fct, target_fct)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size, shuffle=False, collate_fn=collate_fn)
    test_dl = None
    if df_test is not None:
        test_ds = DatasetPceForGNNsimple(df_test, mol2seq, smiles_fct, target_fct)
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False, collate_fn=collate_fn)
    
    batch_func = (lambda dl : len(dl) if dl else 0)
    batch_numbers = list(map(batch_func, [train_dl, val_dl, test_dl]))
    log('batch numbers - train val test=', batch_numbers)
    
    return train_dl, val_dl, test_dl

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

class GNNSimple(oscml.utils.util_lightning.CARESModule):
    
    def __init__(self, args, target_mean, target_std):
        learning_rate = args['LEARNING_RATE']
        super().__init__(learning_rate, target_mean, target_std)
                
        node_type_number = args['NODE_TYPE_NUMBER']
        conv_dim_list = args['CONV_DIM_LIST']
        mlp_dim_list = args['MLP_DIM_LIST']
        if 'PADDING_INDEX' in args and (args['PADDING_INDEX'] is not None):
            self.padding_index = args['PADDING_INDEX']
            log('padding index for embedding was set to ', self.padding_index, '. Thus unknown atom types can be handled')
            # consider the padding index for subsequential transfer learning with unknown atom types:
            # we add +1 to node_type_number because
            # padding_idx = 0 in a sequences is mapped to zero vector
            self.embedding = nn.Embedding(node_type_number+1, conv_dim_list[0], padding_idx=self.padding_index)
        else:
            self.padding_index = None
            log('No padding index for embedding was set. No transfer learning for unknown atom types will be possible')
            self.embedding = nn.Embedding(node_type_number, conv_dim_list[0])
        
        self.conv_modules = nn.ModuleList()
        for i in range(len(conv_dim_list)-1):
            layer = GNNSimpleLayer(conv_dim_list[i], conv_dim_list[i+1], F.relu)
            self.conv_modules.append(layer)
            
        self.mlp = util_pytorch.create_mlp(mlp_dim_list)
        
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