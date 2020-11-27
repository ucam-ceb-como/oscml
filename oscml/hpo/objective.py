import logging

import glob
import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import oscml.data.dataset
import oscml.data.dataset_cep
import oscml.data.dataset_hopv15
import oscml.hpo.hpo_attentivefp
import oscml.hpo.hpo_bilstm
import oscml.hpo.hpo_simplegnn
import oscml.hpo.hpo_rf
import oscml.hpo.hpo_svr
import oscml.hpo.optunawrapper
from oscml.utils.util_config import set_config_param
import oscml.utils.util_sklearn

class MetricsCallback(pl.Callback):

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)

def fit_or_test(model, train_dl, val_dl, test_dl, training_params,
                log_dir, trial=None, trial_number=-1, n_trials=0, cv_index=''):

    epochs = training_params['epochs']
    metric = training_params['metric']
    direction = training_params['direction']
    patience = training_params['patience']

    # create callbacks for Optuna for receiving the metric values from Lightning and for
    # pruning trials
    metrics_callback = MetricsCallback()
    callbacks = [metrics_callback]
    #if trial:
    #    pruning_callback = optuna.integration.PyTorchLightningPruningCallback(trial, monitor=metric)
    #    callbacks.append(pruning_callback)

    if patience > 0:
        early_stopping_callback = EarlyStopping(monitor=metric, min_delta=0.0, patience=patience, verbose=False, mode=direction)
        callbacks.append(early_stopping_callback)

    if cv_index == 'retrain' or cv_index == '':
        dirpath = log_dir + '/trial_' + str(trial_number) + '/' + cv_index + '/'
        checkpoint_callback = ModelCheckpoint(monitor=metric, dirpath=dirpath.replace('//', '/'),
                                              filename=str(cv_index)+'_model',
                                              save_top_k=1, mode=direction[0:3])
        callbacks.append(checkpoint_callback)

    logging.info('[%s] model for trial %s=%s', cv_index, trial_number, model)

    # create standard params for Ligthning trainer

    # if the number of trials is 1 then save checkpoints for the last and best epoch
    # otherwise if HPO is running (i.e. unspecified time-contrained number of trials or finite number > 1 )
    # then save no checkpoints
    trainer_params = oscml.utils.util_lightning.get_standard_params_for_trainer(metric, False)

    # create Lightning metric logger that logs metric values for each trial in its own csv file
    # version='' means that no version-subdirectory is created
    csv_logger = pl.loggers.CSVLogger(save_dir=log_dir,
                                      name='trial_' + str(trial_number),
                                      version=cv_index)

    # put all trainer params together
    trainer_params.update({
        'max_epochs': epochs,
        'logger': csv_logger,
        'callbacks': callbacks
    })

    logging.info('[%s] params for Lightning trainer=%s', cv_index, trainer_params)

    trainer = pl.Trainer(**trainer_params)

    if epochs > 0:
        logging.info('[%s] fitting trial %s / %s', cv_index, trial_number, n_trials)
        trainer.fit(model, train_dataloader=train_dl, val_dataloaders=val_dl)

        # return the value for the metric specified in the start script
        if patience > 0:
            # return the best score while early stopping is applied
            val_error = early_stopping_callback.best_score.item()
        else:
            val_error = metrics_callback.metrics[-1][metric].item()

        logging.info('[%s] finished fitting for trial %s with %s = %s', cv_index, trial_number, metric, val_error)

    if test_dl:
        if cv_index == 'retrain' or cv_index == '':
            ckpt_path = glob.glob(dirpath+str(cv_index)+'_model' + '*.ckpt')[0].replace('\\', '/')
            model.load_state_dict(torch.load(ckpt_path)['state_dict'])
            model.eval()
            test_result = trainer.test(model, test_dataloaders=test_dl)[0]
            logging.info('[%s] result=%s', cv_index, test_result)

    if epochs > 0:
        return val_error
    return test_result


def get_training_params(trial, training_settings):
    training_params = {}
    for key, value in training_settings.items():
        if key == 'optimiser':
            optimiser = {}
            for opt_key, opt_value in training_settings[key].items():
                optimiser.update({opt_key: set_config_param(trial=trial,param_name=opt_key,param=opt_value, all_params=optimiser)})
            training_params[key] = optimiser
        else:
            training_params[key] = set_config_param(trial=trial,param_name=key,param=value, all_params=training_params)
    return training_params


def get_model_and_data(model_name, trial, config, df_train, df_val, df_test, training_params, transformer, log_dir):
    if model_name == 'BILSTM':
        return oscml.hpo.hpo_bilstm.create(trial, config, df_train, df_val, df_test, training_params['optimiser'], transformer)
    elif model_name == 'AttentiveFP':
        return oscml.hpo.hpo_attentivefp.create(trial, config, df_train, df_val, df_test, training_params['optimiser'], transformer, log_dir)
    elif model_name == 'SimpleGNN':
        return oscml.hpo.hpo_simplegnn.create(trial, config, df_train, df_val, df_test, training_params['optimiser'], transformer)
    return None


def objective(trial, config, df_train, df_val, df_test, transformer, log_dir, total_number_trials):

    # init parameters from config file
    model_name = config['model']['name']
    training_params = get_training_params(trial, config['training'])
    cv = config['training']['cross_validation']
    seed = config['numerical_settings']['seed']

    # deal with RF and SVR models first
    if model_name == 'RF':
        model, x_train, y_train, x_val, y_val, x_test, y_test = oscml.hpo.hpo_rf.create(trial, config, df_train, df_val, df_test, training_params)
        metric_value = oscml.utils.util_sklearn.train_and_test(x_train, y_train, x_val, y_val, x_test, y_test, model,
                                                               training_params['cross_validation'],
                                                               training_params['criterion'])
    elif model_name == 'SVR':
        model, x_train, y_train, x_val, y_val, x_test, y_test = oscml.hpo.hpo_svr.create(trial, config, df_train, df_val, df_test, training_params)
        metric_value = oscml.utils.util_sklearn.train_and_test(x_train, y_train, x_val, y_val, x_test, y_test, model,
                                                               training_params['cross_validation'],
                                                               training_params['criterion'])

    # then move to BILSTM, AttentiveFP, and SimpleGNN models
    elif model_name == 'BILSTM' or model_name == 'AttentiveFP' or model_name == 'SimpleGNN':
        trial_number = trial.number
        # apply cross-validation
        if isinstance(cv, int) and cv > 1:
            kf = KFold(n_splits=cv, random_state=seed, shuffle=True)
            assert df_val is None, "validation set should be added to training set for cross validation"
            kf.get_n_splits(df_train)
            cv_index = 1
            cv_metric = []
            for train_index, val_index in kf.split(df_train):
                logging.info('run %s of %s fold cross-validation', cv_index, cv)
                model, train_dl, val_dl, test_dl = get_model_and_data(model_name, trial, config,
                                                                      df_train.iloc[train_index], df_train.iloc[val_index],
                                                                      df_test, training_params, transformer, log_dir)
                metric_value = fit_or_test(model, train_dl, val_dl, test_dl, training_params, log_dir,
                                           trial, trial_number, total_number_trials, str(cv_index))
                cv_index += 1
                cv_metric.append(metric_value)
            metric_value = np.array(cv_metric).mean()

            # retrain model over the whole training and validation dataset (with random spliting of validation)
            rs = ShuffleSplit(n_splits=1, test_size=0.20, random_state=seed+1)
            rs.get_n_splits(df_train)
            for retrain_index, reval_index in rs.split(df_train):

                model, train_dl, val_dl, test_dl = get_model_and_data(model_name, trial, config,
                                                                      df_train.iloc[retrain_index], df_train.iloc[reval_index],
                                                                      df_test, training_params, transformer, log_dir)
                fit_or_test(model, train_dl, val_dl, test_dl, training_params, log_dir,
                            trial, trial_number, total_number_trials, 'retrain')

        # normal training and testing
        else:
            model, train_dl, val_dl, test_dl = get_model_and_data(model_name, trial, config, df_train, df_val, df_test,
                                                                  training_params, transformer, log_dir)
            metric_value = fit_or_test(model, train_dl, val_dl, test_dl, training_params, log_dir,
                                       trial, trial_number, total_number_trials, '')

    else:
        return None

    return metric_value
