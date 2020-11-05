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

import oscml.data.dataset
import oscml.data.dataset_hopv15
import oscml.models.model_example_mlp_mnist
import oscml.start
import oscml.test_oscml
import oscml.utils.params
from oscml.utils.params import cfg
import oscml.utils.util
from oscml.utils.util import concat
import oscml.utils.util_lightning
import oscml.utils.util_pytorch

def process(src, dst, epochs, csv_logger):

        # define data loaders and params
        data_loader_fct = oscml.models.model_example_mlp_mnist.get_mnist

        data_loader_params = {
            'mnist_dir': './tmp', 
            'batch_size': 128
        }


        # define models and params
        model = oscml.models.model_example_mlp_mnist.MlpWithLightning
        
        model_params =  {
            # model parameters
            'number_classes': 10, 
            'layers': 3,
            'units': [100, 50, 20], 
            'dropouts': [0.2, 0.2, 0.2],
            # optimizer parameters
            'optimizer': 'Adam', 
            'optimizer_lr': 0.0015
        }


        # define params for Lightning trainer
        #trainer_params = oscml.utils.util_lightning.get_standard_params_for_trainer_short()
        trainer_params = oscml.utils.util_lightning.get_standard_params_for_trainer(monitor='val_acc')
        trainer_params.update({
            'max_epochs': epochs,
            'logger': csv_logger
        })

        
        # create the model, dataloaders and Lightning trainer
        model_instance = model(**model_params)
        train_dl, val_dl = data_loader_fct(**data_loader_params)
        trainer = pl.Trainer(**trainer_params)


        # train the model
        trainer.fit(model_instance, train_dataloader=train_dl, val_dataloaders=val_dl)


        # test
        logging.info('start testing')
        # the mnist example didn't provide a test test. Thus, we will use
        # the validation dataloader as test loader here.
        test_dl = val_dl
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
        model, model_instance, trainer, test_dl = process(src, dst, epochs, csv_logger)
    except BaseException as exc:
        print(exc)
        logging.exception('finished with exception', exc_info=True)
        raise exc
    else:
        logging.info('finished successfully')
    
    return model, model_instance, trainer, test_dl


if __name__ == '__main__':
    print('current working directory=', os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default='.')
    parser.add_argument("--dst", type=str, default='.')
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    start(args.src, args.dst, args.epochs)