import datetime

import numpy as np
import pytorch_lightning as pl
import sklearn
import torch
import torch.optim
import torch.nn

from oscml.utils.util import log, calculate_metrics

def get_standard_params_for_trainer(root_dir='./'):
    save_dir = root_dir + 'logs'
    #tb_logger = pl.loggers.TensorBoardLogger(save_dir='save_dir, name='tb')
    csv_logger = pl.loggers.CSVLogger(save_dir=save_dir, name='csv')


    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss', save_last=True, period=2, save_top_k=-1)

    # does not work at the moment on laptop with gpu
    #gpus = 1 if torch.cuda.is_available() else 0
    gpus = 0

    params = {
        'log_every_n_steps': 1,
        'flush_logs_every_n_steps': 10,
        'progress_bar_refresh_rate': 1,
        'logger': csv_logger, #[csv_logger, tb_logger],
        'checkpoint_callback': checkpoint_callback,
        'gpus': gpus,
    }

    log('params for Lightning trainer=', params)
    log('trainer is logging to save_dir=', csv_logger.save_dir, #', name=', csv_logger.name,
        ', experiment version=', csv_logger.version)

    return params

class CARESModule(pl.LightningModule):

    def __init__(self, learning_rate, target_mean, target_std):
        super().__init__()

        log('initializing CARESModule with learning_rate=', learning_rate, ', target_mean=', target_mean, ', target_std=', target_std)

        self.learning_rate = learning_rate
        self.target_mean = target_mean
        self.target_std = target_std
        if self.target_mean:
            self.inverse_transform_fct = lambda x : x * target_std + target_mean
        else:
            self.inverse_transform_fct = None

        self.test_predictions = None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_count', len(y))
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
        result = {
            'epoch': epoch,
            'time': str(datetime.datetime.now()),
            'loss': loss}
        prefix = 'val'
    else: # test
        result = {'loss': loss}
        prefix = 'test'
    result.update(metrics)

    result_with_prefix = {}
    for key, value in result.items():
        result_with_prefix[prefix + '_' + key] = value

    log(prefix + ' ' + 'result=', result_with_prefix)

    return (result_with_prefix, y_complete, y_hat_complete)