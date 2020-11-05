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
from oscml.utils.util import log
from oscml.utils.util import smiles2mol
import oscml.utils.util_lightning as util_lightning
import oscml.utils.util_pytorch
import oscml.features.weisfeilerlehman

def DEPRECATED_init_params(df, mol2seq=None):
    log('initializing parameters for PyTorch model BiLSTM')
    d = {}
    cfg['BILSTM'] = d
    if not mol2seq:
        mol2seq = oscml.features.weisfeilerlehman.mol2seq(1, df)
    d['MOL2SEQ'] = mol2seq
    d['SUBGRAPH_NUMBER'] = len(mol2seq.fragment_dict)
    d['SUBGRAPH_EMBEDDING_DIM'] = 128
    d['MAX_SEQUENCE_LENGTH'] = 60 # max number of atoms = 52
    d['BATCH_SIZE'] = 250
    d['OUTPUT_DIM'] = 1
    d['MLP_DIM_LIST'] = [256, 32, 32, 32, d['OUTPUT_DIM']]
    d['LEARNING_RATE'] = 0.001
    d['PADDING_INDEX'] = 0
    #log('subgraph_number=', d['SUBGRAPH_NUMBER'])
    #log('max sequence length=', d['MAX_SEQUENCE_LENGTH'])
    log('parameters for PyTorch model BiLSTM:', d)

class DatasetForBiLstmWithTransformer(torch.utils.data.Dataset):
    
    def __init__(self, df, max_sequence_length, m2seq_fct, padding_index, smiles_fct, target_fct):
        super().__init__()
        self.df = df
        self.max_sequence_length = max_sequence_length
        self.m2seq_fct = m2seq_fct
        self.padding_sequence = [padding_index]*100

        if isinstance(smiles_fct, str):
            self.smiles_fct = lambda data: data[smiles_fct]
        else:
            self.smiles_fct = smiles_fct
        if isinstance(target_fct, str):
            self.target_fct = lambda data: data[target_fct]
        else:
            self.target_fct = target_fct
    
    def __getitem__(self, index):
        #smiles = self.df.iloc[index]['SMILES_str']
        row = self.df.iloc[index]
        smiles = self.smiles_fct(row)
        m = smiles2mol(smiles)
        x = self.m2seq_fct(m)
        # increase all indexes by +1
        x = np.array(x) + 1
        # fill up the sequence with padding index 0
        x = np.append(x, self.padding_sequence[:self.max_sequence_length-len(x)])
        device = cfg[oscml.utils.params.PYTORCH_DEVICE]
        x = torch.as_tensor(x, dtype = torch.long, device = device)

        #y = torch.as_tensor(np.array(self.df.iloc[index]['pcez'], dtype=np.float32), device = device)
        y = self.target_fct(row)
        y = torch.as_tensor(np.array(y, dtype=np.float32), device = device)

        return [x, y]
    
    def __len__(self):
        return len(self.df)

def get_dataloaders(train, val, test, batch_size, mol2seq, max_sequence_length, padding_index,
        smiles_fct='SMILES_str', target_fct='pcez'):

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
    log('batch numbers - train val test=', batch_numbers)
    
    if test is None:
        return train_dl, val_dl 
    return train_dl, val_dl, test_dl
    
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
    
    def __init__(self, number_of_subgraphs, subgraph_embedding_dim, mlp_dim_list, padding_index, target_mean, target_std, learning_rate):

        super().__init__('Adam', learning_rate, target_mean, target_std)
        self.save_hyperparameters()
        
        assert len(mlp_dim_list) > 0
        # since the hidden state vectors of both LSTMs have the same dimension
        # and are concatenated, the number of neurons in the first MLP layer
        # must be even
        assert mlp_dim_list[0] % 2 == 0 
        
        # we add +1 to number_of_subgraphs because
        # padding_idx = 0 in a sequences is mapped to zero vector
        self.embedding = nn.Embedding(number_of_subgraphs+1, subgraph_embedding_dim, padding_idx=padding_index)
        lstm_output_dim = int(mlp_dim_list[0] / 2)
        self.bilstm = nn.LSTM(subgraph_embedding_dim, lstm_output_dim, bidirectional=True)
        self.attention = Attention(2 * lstm_output_dim, 2 * lstm_output_dim)
        self.mlp = oscml.utils.util_pytorch.create_mlp(mlp_dim_list)
    
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