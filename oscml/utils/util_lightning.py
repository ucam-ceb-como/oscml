import datetime
import logging

import numpy as np
import pytorch_lightning as pl
import sklearn
import torch
import torch.optim
import torch.nn

from oscml.utils.util import concat, calculate_metrics

def get_standard_params_for_trainer_short():
    params = {
        'max_epochs': 1,
        'log_every_n_steps': 1,
        'flush_logs_every_n_steps': 10,
        'gpus': 1 if torch.cuda.is_available() else None,
        }
    return params

def get_standard_params_for_trainer(metric):
 
    # https://pytorch-lightning.readthedocs.io/en/latest/generated/pytorch_lightning.callbacks.ModelCheckpoint.html
    # By default, dirpath is None and will be set at runtime to the location specified 
    # by Trainer’s default_root_dir or weights_save_path arguments, 
    # and if the Trainer uses a logger, the path will also contain logger name and version.
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor=metric, save_last=True, period=1, save_top_k=10)

    # does not work at the moment on laptop with gpu
    #gpus = 1 if torch.cuda.is_available() else 0
    gpus = 0

    params = {
        'log_every_n_steps': 1,
        'flush_logs_every_n_steps': 10,
        'progress_bar_refresh_rate': 1,
        'checkpoint_callback': checkpoint_callback,
        'gpus': gpus,
    }

    return params

class OscmlModule(pl.LightningModule):

    def __init__(self, optimizer, target_mean, target_std):
        super().__init__()

        params = {'optimizer': optimizer, 'target_mean': target_mean, 'target_std': target_std}
        logging.info(concat('initializing OscmlModule with', params))

        self.optimizer = optimizer
        self.target_mean = target_mean
        self.target_std = target_std
        if self.target_mean:
            self.inverse_transform_fct = lambda x : x * target_std + target_mean
        else:
            self.inverse_transform_fct = None

        self.test_predictions = None

    def configure_optimizers(self):
        #optimizer = getattr(torch.optim, self.optimizer)(self.parameters(), lr=self.optimizer_lr)
        opt_params = self.optimizer.copy()
        name = opt_params.pop('name')
        optimizer_instance = getattr(torch.optim, name)(self.parameters(), **opt_params)
        logging.info('created optimizer=' + str(optimizer_instance))
        return optimizer_instance

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.MSELoss()(y_hat, y)
        self.log('phase', 'train')
        self.log('loss', loss)
        self.log('count', len(y))
        self.log('time', str(datetime.datetime.now()))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return y, y_hat

    def validation_epoch_end(self, outputs):
        result, _, _ = shared_epoch_end(
            outputs, is_validation=True, epoch=self.current_epoch,
            inverse_transform_fct=self.inverse_transform_fct)

        for key, value in result.items():
            self.log(key, value)

        return super().validation_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return y, y_hat

    def test_epoch_end(self, outputs):
        result, y, y_hat = shared_epoch_end(
            outputs, is_validation=False, epoch=self.current_epoch,
            inverse_transform_fct=self.inverse_transform_fct)

        for key, value in result.items():
            self.log(key, value)

        self.test_predictions = (y, y_hat)
        return super().test_epoch_end(outputs)

def shared_epoch_end(tensor_step_outputs, is_validation, epoch, inverse_transform_fct):
    y_complete = np.array([])
    y_hat_complete = np.array([])
    for outputs in tensor_step_outputs:
        y = outputs[0].detach().cpu().numpy()
        y_complete = np.concatenate((y_complete, y))
        y_hat = outputs[1].detach().cpu().numpy()
        y_hat_complete = np.concatenate((y_hat_complete, y_hat))

    loss = sklearn.metrics.mean_squared_error(y_complete, y_hat_complete, squared=True)
    if inverse_transform_fct:
        y_complete = inverse_transform_fct(y_complete)
        y_hat_complete = inverse_transform_fct(y_hat_complete)
    metrics = calculate_metrics(y_complete, y_hat_complete)
    
    if is_validation:
        result = {'phase': 'val', 'val_loss': loss}
    else: # test
        result = {'phase': 'test'}
    
    result.update({
        'epoch': epoch,
        'time': str(datetime.datetime.now()),
    })
    result.update(metrics)
    logging.info(concat('result=', result))

    return (result, y_complete, y_hat_complete)