import argparse

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

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

def preprocess(file_path):
    
    # set random seeds
    seed = 100
    log('numpy seed=', seed)
    np.random.seed(seed)
    seed = 100
    log('torch seed=', seed)
    torch.manual_seed(seed)
    
    # parameters for preprocessing    
    args = {
        'threshold': 0.0001,
        'number_samples': 25000,
        'train_ratio': 0.6,
        'val_ratio': 0.2,
        'test_ratio': 0.2}
    
    # cleaned, split and normalized data as pandas dataframes
    return oscml.data.dataset_cep.preprocess_CEP(file_path, **args)
    
def train_bilstm(file_path, epochs, plot):
    
    transformer, df_train, df_val, df_test, df_train_plus_val, df_train_plus_val_plus_test = \
        preprocess(file_path)
    
    # parameters for the BiLSTM model (including fragments from the Weisfeiler Lehman algorithm)
    #model_bilstm.init_params(df_train_plus_val_plus_test)
    mol2seq = oscml.data.dataset_cep.mol2seq_precalculated_with_OOV(None, radius=1, oov=True)
    oscml.models.model_bilstm.init_params(df_train_plus_val_plus_test, mol2seq)
    log('all params=\n', cfg)
    args = cfg['BILSTM']
    
    # convert dataframes into Pytorch's dataloader  
    train_dl, val_dl, test_dl = oscml.models.model_bilstm.get_dataloaders(df_train, df_val, df_test, args)
    
    # create the model
    mean = transformer.target_mean
    std = transformer.target_std
    model = oscml.models.model_bilstm.BiLstmForPce(args, mean, std)
    #model.to(cfg['PYTORCH_DEVICE'])
    log('model=\n', model)
    
    # train the model
    params_trainer = oscml.utils.util_lightning.get_standard_params_for_trainer(root_dir = '../')
    trainer = pl.Trainer(max_epochs=epochs, **params_trainer)
    trainer.fit(model, train_dl, val_dl)
        
    # test the model
    metrics = trainer.test(model, test_dataloaders=test_dl) 

    if plot:
        # works in Jupyter notebook 
        # when starting in anaconda environment, a window with the plot will pop up
        y, y_hat = model.test_predictions
        oscml.visualization.util_plot.plot(y, y_hat)

def train_gnnsimple(file_path, epochs, plot):
    
    transformer, df_train, df_val, df_test, df_train_plus_val, df_train_plus_val_plus_test = \
        preprocess(file_path)
        
    # parameters for the GNNSIMPLE model
    # use the default node mapping
    #model_gnn.init_params(df_train_plus_val_plus_test, dataset_cep.ATOM_TYPES_CEP)
    oscml.models.model_gnn.init_params(None, oscml.data.dataset_cep.ATOM_TYPES_CEP)
    log('all params=\n', cfg)
    args = cfg['GNNSIMPLE']
    
    # convert dataframes into Pytorch's dataloader  
    train_dl, val_dl, test_dl = oscml.models.model_gnn.get_dataloaders(df_train, df_val, df_test, args)
    
    # create the model
    mean = transformer.target_mean
    std = transformer.target_std
    model = oscml.models.model_gnn.GNNSimple(args, mean, std)
    #model.to(cfg['PYTORCH_DEVICE'])
    log('model=\n', model)
    
    # train the model
    params_trainer = oscml.utils.util_lightning.get_standard_params_for_trainer(root_dir = '../')
    trainer = pl.Trainer(max_epochs=epochs, **params_trainer)
    trainer.fit(model, train_dl, val_dl)
        
    # test the model
    metrics = trainer.test(model, test_dataloaders=test_dl) 

    if plot:
        # works in Jupyter notebook 
        # when starting in anaconda environment, a window with the plot will pop up
        y, y_hat = model.test_predictions
        oscml.visualization.util_plot.plot(y, y_hat)
    
def main(args):
    if args.task == 'skipinvalidsmiles':
        oscml.data.dataset_cep.store_CEP_with_valid_SMILES(args.src, args.dest, args.numbersamples)
    elif args.task == 'train':
        if not args.model or args.model == 'bilstm':
            train_bilstm(args.src, args.epochs, args.plot)
        elif args.model == 'gnnsimple':
            train_gnnsimple(args.src, args.epochs, args.plot)
        else:
            log('unknown model=', args.model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CEP')
    parser.add_argument("--task", choices=['skipinvalidsmiles', 'train'])
    parser.add_argument("--model", type=str)
    parser.add_argument("--src", type=str)
    parser.add_argument("--dest", type=str)
    parser.add_argument("--numbersamples", type=int)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--plot", type=bool, default=False)
    
    args = parser.parse_args()
    
    #__init__.init()
    log('parameters=', args)
    main(args)