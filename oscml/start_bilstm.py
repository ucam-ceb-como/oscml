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
import oscml.start
import oscml.utils.params
from oscml.utils.params import cfg
import oscml.utils.util
from oscml.utils.util import concat
import oscml.utils.util_lightning
import oscml.utils.util_pytorch
import oscml.visualization.util_plot

def process(src, dst, epochs, csv_logger):
   
    # read data and preprocess, e.g. standarization, splitting into train, validation and test set
    path = oscml.start.path_cepdb_25000(src)
    df_train, df_val, df_test = oscml.data.dataset.read_and_split(path)
    df_train = df_train[:1500].copy()
    df_val = df_val[:500].copy()
    df_test = df_test[:500].copy()


    # define data loader and params
    data_loader_fct = oscml.models.model_bilstm.get_dataloaders

    transformer = oscml.data.dataset.create_transformer(df_train, column_target='pce', column_x='SMILES_str')
    mol2seq = oscml.data.dataset_cep.mol2seq_precalculated_with_OOV(None, radius=1, oov=True)

    data_loader_params = {
        'train': df_train,
        'val': df_val,
        'test': None, 
        'batch_size': 250, 
        'mol2seq': mol2seq,
        'max_sequence_length': 60,
        'padding_index': 0,
        'smiles_fct': transformer.transform_x,
        'target_fct': transformer.transform, 
    }


    # define models and params
    model = oscml.models.model_bilstm.BiLstmForPce

    model_params =  {
        'number_of_subgraphs': 60,
        'subgraph_embedding_dim': 128,
        'mlp_dim_list': [256, 32, 32, 32, 1],
        'padding_index': 0,
        'target_mean': transformer.target_mean, 
        'target_std': transformer.target_std,
        'learning_rate': 0.001,
    }


    # define params for Lightning trainer
    trainer_params = oscml.utils.util_lightning.get_standard_params_for_trainer(monitor='val_loss')
    trainer_params.update({
        'max_epochs': epochs,
        'logger': csv_logger
    })


    # put all information together
    params = {
        'data_loader_fct': data_loader_fct,
        'data_loader_params': data_loader_params,
        'model': model,
        'model_params': model_params,
        'trainer_params': trainer_params
    }


    # train
    model_instance, trainer = oscml.utils.util_lightning.fit_model(**params)


    # test
    logging.info('start testing')
    data_loader_params['test'] = df_test
    _, _, test_dl = data_loader_fct(**data_loader_params)
    metrics = trainer.test(model_instance, test_dataloaders=test_dl)
    logging.info(metrics)

    return model, model_instance, trainer, test_dl


def start(src, dst, epochs):
   # initialize, e.g. logging
    csv_logger = oscml.utils.util.init_logging(src, dst)
    logging.info('current working directory=' + os.getcwd())
    logging.info(concat('src=', src, ', dst=', dst, ', epochs=', epochs))

    np.random.seed(200)
    torch.manual_seed(200)

    try:
        return process(src, dst, epochs, csv_logger)
    except BaseException as exc:
        print(exc)
        logging.exception('finished with exception', exc_info=True)
        raise exc
    else:
        logging.info('finished successfully')
    

if __name__ == '__main__':
    print('current working directory=', os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default='.')
    parser.add_argument("--dst", type=str, default='.')
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    start(args.src, args.dst, args.epochs)