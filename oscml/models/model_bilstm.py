import collections
import logging

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import oscml.utils.params
from oscml.utils.params import cfg
from oscml.utils.util import smiles2mol, concat
import oscml.utils.util_lightning as util_lightning
import oscml.utils.util_pytorch
import oscml.features.weisfeilerlehman


class Mol2seq():
    
    def __init__(self, radius, oov, wf=None):

        self.radius = radius
        self.oov = oov
        self.wf = wf

        if wf:
            atom_dict = wf['atom_dict']
            bond_dict = wf['bond_dict']
            fragment_dict = wf['fragment_dict']
            edge_dict = wf['edge_dict']

        self.atom_dict = collections.defaultdict(lambda:len(self.atom_dict), atom_dict)
        self.bond_dict = collections.defaultdict(lambda:len(self.bond_dict), bond_dict)
        self.fragment_dict = collections.defaultdict(lambda:len(self.fragment_dict), fragment_dict)
        self.edge_dict = collections.defaultdict(lambda: len(self.edge_dict), edge_dict)
    
        # fragment index starts with 0, thus -1
        self.max_index = len(self.fragment_dict) - 1
        logging.info(concat('initialized Mol2Seq with radius=', radius, ', oov=', oov, ', max_index=', self.max_index))
        
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


class DatasetForBiLstmWithTransformer(torch.utils.data.Dataset):
    
    def __init__(self, df, max_sequence_length, m2seq_fct, padding_index, smiles_fct, target_fct):
        super().__init__()
        self.df = df
        self.max_sequence_length = max_sequence_length
        self.m2seq_fct = m2seq_fct
        # TODO: use torch.nn.utils.rnn.pack_padded_sequence instead
        self.padding_sequence = [padding_index]*self.max_sequence_length

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
        m = smiles2mol(smiles)
        x = self.m2seq_fct(m)
        # increase all indexes by +1
        x = np.array(x) + 1
        # fill up the sequence with padding index 0
        diff = self.max_sequence_length-len(x)
        if diff > 0:
            x = np.append(x, self.padding_sequence[:diff])
        if diff < 0:
            raise RuntimeError(concat('A sequence with length greater the maximum sequence length was generated.',
                    ', length=', len(x), ', maximum sequence length=', self.max_sequence_length, ', row index=', str(index)))

        device = cfg[oscml.utils.params.PYTORCH_DEVICE]
        x = torch.as_tensor(x, dtype = torch.long, device = device)

        y = self.target_fct(row)
        y = torch.as_tensor(np.array(y, dtype=np.float32), device = device)

        return [x, y]
    
    def __len__(self):
        return len(self.df)

def get_dataloaders_internal(train, val, test, batch_size, mol2seq, max_sequence_length, smiles_fct, target_fct):

    padding_index = 0

    train_dl = None
    if train is not None:
        train_ds = DatasetForBiLstmWithTransformer(train, max_sequence_length, mol2seq, padding_index, smiles_fct, target_fct)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True)
    val_dl = None
    if val is not None:
        val_ds = DatasetForBiLstmWithTransformer(val, max_sequence_length, mol2seq, padding_index, smiles_fct, target_fct)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size, shuffle=False)
    test_dl = None
    if test is not None:
        test_ds = DatasetForBiLstmWithTransformer(test, max_sequence_length, mol2seq, padding_index, smiles_fct, target_fct)
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False)
   
    batch_func = (lambda dl : len(dl) if dl else 0)
    batch_numbers = list(map(batch_func, [train_dl, val_dl, test_dl]))
    logging.info('batch numbers - train val test=' + str(batch_numbers))
    
    return train_dl, val_dl, test_dl


def get_dataloaders(dataset, df_train, df_val, df_test, transformer, batch_size, max_sequence_length):

    info = oscml.data.dataset.get_dataset_info(dataset)
    mol2seq = info.mol2seq
    return get_dataloaders_internal(df_train, df_val, df_test, batch_size, mol2seq, max_sequence_length, 
                smiles_fct = transformer.transform_x,
                target_fct = transformer.transform)


class Attention(pl.LightningModule):
    
    def __init__(self, vector_dim, sub_fragment_context_vector_dim):
        super().__init__()
        
        # equation (11) - (13) from paper [Wu20]
        # "u_sub is the sub-fragment context vector. 
        # It is randomly initialized and 
        # jointly learned during the network training process, 
        # so are the other involved vectors, W_sub and b_sub"
 
        # W_sub and b_sub from eq. (11)
        self.linear = nn.Linear(vector_dim, sub_fragment_context_vector_dim)
        # tanh from eq. (11)
        self.tanh = nn.Tanh()
        # dot product from eq. (12)
        
        initialized_u_sub_tensor = torch.nn.init.normal_(torch.Tensor(sub_fragment_context_vector_dim))
        self.u_sub = nn.parameter.Parameter(initialized_u_sub_tensor)
        # softmax from eq. (12) over the dim t ('time') of the sequence 
        # the sum expression of eq. (12) is not correct, it should be
        #     sum_i exp(u_i^T * u_s)    
        # i.e. u_i^T instead of u_t^T
        self.softmax = nn.Softmax(dim=1) 
    
    def forward(self, h):
        
        # example: h with batch size 2, max sequence length 3 and vector dim 256
        # h.size() = [2, 3, 256]
        #print('MY ATT INPUT H', h.size(), h)
        
        # just for ease, we assume vector_dim = sub_fragment_context_vector_dim,
        # i.e. the matrix W_sub has quadratic shape [256, 256] and b_sub has shape [256]
        x = self.linear(h)
        
        # tanh is applied element-wise and doesn't change the shape
        x = self.tanh(x)
        # x.size() = [2, 3, 256]
        #print('MY ATT TANH', x.size(), x)
        
        # u_sub.size() = [256]
        #print('MY AT U_SUB', self.u_sub.size(), self.u_sub)
        
        # dot product u_t * u_sub from equation (12):
        # function torch.dot is not capable for broadcasting and batch-processing
        # thus we have break the dot product into two separate operations:
        # a) element-wise multiplication with broadcasting --> size = [2, 3, 256]
        # b) sum along the last dimension --> size = [2,3]
        x = torch.sum(self.u_sub * x, dim=2)
        #print('MY ATT DOT ', x.size(), x)
        
        # softmax respects batch-processing --> alpha.size() = [2,3]
        # i.e. alpha contains two attention vectors with probability weights for t=0,1,2 as elements
        # example output: 
        #      MY ATT ALPHA torch.Size([2, 3]) tensor([[0.3276, 0.3442, 0.3282],
        #      [0.3297, 0.3471, 0.3231]], grad_fn=<SoftmaxBackward>)
        alpha = self.softmax(x)
        #print('MY ATT ALPHA', alpha.size(), alpha)
      
        # unsqueeze(dim=2) means: add an "empty" dim at the end --> [2,3,1]
        # elementwise multiplication with the orinal h from the input --> [2, 3, 256]
        m = alpha.unsqueeze(dim=2) * h    
        #print('MY ATT M', m.size(), m)
        
        # sum along sequence index t, i.e. along the dim=1 --> [2,256]
        msum = torch.sum(m, dim=1)
        #print('MY ATT MSUM', msum.size(), msum)
        
        return msum
    
class BiLstmForPce(util_lightning.OscmlModule):
    
    def __init__(self, number_of_subgraphs, subgraph_embedding_dim, lstm_hidden_dim, mlp_units, padding_index, target_mean, target_std, optimizer, mlp_dropouts=None):

        super().__init__(optimizer, target_mean, target_std)
        logging.info('initializing ' + str(locals()))

        self.save_hyperparameters()
        
        assert len(mlp_units) > 0
        
        # we add +1 to number_of_subgraphs because
        # padding_idx = 0 in a sequences is mapped to zero vector
        self.embedding = nn.Embedding(number_of_subgraphs+1, subgraph_embedding_dim, padding_idx=padding_index)
        self.bilstm = nn.LSTM(input_size=subgraph_embedding_dim, hidden_size=lstm_hidden_dim, bidirectional=True)
        # factor 2 because the LSTM is birectional
        lstm_output_dim = 2 * lstm_hidden_dim
        self.attention = Attention(lstm_output_dim, lstm_output_dim)

        mlp_units_with_input_dim = [lstm_output_dim]
        mlp_units_with_input_dim.extend(mlp_units)
        self.mlp = oscml.utils.util_pytorch.create_mlp(mlp_units_with_input_dim, mlp_dropouts)

    
    def forward(self, index_sequences):
        
        # get the sequences of embedding vectors corresponding to subgraph indexes
        x = self.embedding(index_sequences)
        
        # h is the sequence of hidden state vectors; 
        # the state vectors from both LSTMs are already concatenated
        h, _ = self.bilstm(x)
        # h has shape [batch size, input sequence length, 256]
        # where 256 = lstm_output_dim = 2 * hidden state vector dim = number neurons in the first MLP layer
        #print('MY H', h.size())
        
 
       
        # 1. sum
        # attention mechanism, here only sum (i.e. multiplication with fixed attention weights = 1)*
        # sum_n=1..60 h_n / 60 --> one vector of size [256]
        # --> [batch size, 256]
        #attention_value_msum = torch.mean(h, dim=1)
        
        # 2. equation (11) - (13) from paper [Wu20]
        attention_value_msum = self.attention(h)
        # --> [batch size, 256]
        #print('MY MSUM', attention_value_msum.size(), attention_value_msum)
        
        o = self.mlp(attention_value_msum)
        # transform from size [batch_size, 1] to [batch_size]
        o = o.view(len(o))

        return o