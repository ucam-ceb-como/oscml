import argparse
import logging
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import numpy as np

import oscml.data.dataset_cep
import oscml.models.model_bilstm
import oscml.models.model_gnn
import oscml.utils.params
from oscml.utils.params import cfg
import oscml.utils.util
from oscml.utils.util import log
import oscml.utils.util_lightning
import oscml.utils.util_pytorch
import oscml.visualization.util_plot

def path_cepdb_valid_smiles(root='.'):
    return root + '/data/processed/CPEDB_valid_SMILES.csv'

def path_cepdb_25000(root='.'):
    return root + '/data/processed/CEPDB_25000.csv'

def path_hopv_15(root='.'):
    return root + '/data/raw/HOPV_15_revised_2.data'

def path_osaka(root='.'):
    return root + '/data/raw/Nagasawa_RF_SI.txt'

def main(args):
    if args.task == 'skipinvalidsmiles':
        oscml.data.dataset_cep.store_CEP_with_valid_SMILES(args.src, args.dest, args.numbersamples)

if __name__ == '__main__':

    print('current working directory=', os.getcwd())

    parser = argparse.ArgumentParser(description='CEP')
    parser.add_argument("--task", choices=['skipinvalidsmiles'])
    parser.add_argument("--src", type=str, default='./data/CPEDB_valid_SMILES.csv')
    parser.add_argument("--dest", type=str)
    parser.add_argument("--numbersamples", type=int)

    args = parser.parse_args()

    log('parameters=', args)
    main(args)