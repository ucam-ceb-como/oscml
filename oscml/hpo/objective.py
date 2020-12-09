import logging

import glob
import torch
import numpy as np
import pandas as pd
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
import oscml.visualization.util_sns_plot

class MetricsCallback(pl.Callback):

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


def standard_score_transform(transformer, y):
    y_transform = (y - transformer.target_mean) / transformer.target_std
    return y_transform


def fit_or_test(model, train_dl, val_dl, test_dl, training_params,
                log_dir, trial=None, trial_number=-1, n_trials=0, cv_index='', best_trial_retrain=False,
                transformer=None, inverse=False, regression_plot=False):
    # TODO investigate the discrepancy between best_trial and retrain, related to transformer? continued training?
    epochs = training_params['epochs']
    metric = training_params['metric']
    direction = training_params['direction']
    patience = training_params['patience']
    min_delta = training_params['min_delta']

    if best_trial_retrain:
        log_head = '[Best trial retrain - Trial ' + str(trial_number) + ']'
    else:
        if cv_index == '':
            log_head = '[Trial ' + str(trial_number) + ']'
        else:
            log_head = '[Trial '+ str(trial_number) + ' - fold ' + str(cv_index) + ']'

    # create callbacks for Optuna for receiving the metric values from Lightning and for
    # pruning trials
    metrics_callback = MetricsCallback()
    callbacks = [metrics_callback]
    #if trial:
    #    pruning_callback = optuna.integration.PyTorchLightningPruningCallback(trial, monitor=metric)
    #    callbacks.append(pruning_callback)

    if patience > 0:
        early_stopping_callback = EarlyStopping(monitor=metric, min_delta=min_delta, patience=patience, verbose=False, mode=direction)
        callbacks.append(early_stopping_callback)

    # only save model checkpoint in the retraining phase, which is only set true for the best trial
    if best_trial_retrain:
        dirpath = log_dir + '/trial_' + str(trial_number) + '/'
        checkpoint_callback = ModelCheckpoint(monitor=metric, dirpath=dirpath.replace('//', '/'),
                                              filename='best_trial_retrain_model',
                                              save_top_k=1, mode=direction[0:3])
        callbacks.append(checkpoint_callback)

    logging.info('%s model for trial %s=%s', log_head, trial_number, model)

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
        'callbacks': callbacks,
        'profiler': 'simple'
    })

    logging.info('%s params for Lightning trainer=%s', log_head, trainer_params)

    trainer = pl.Trainer(**trainer_params)

    if epochs > 0:
        logging.info('%s fitting trial %s / %s', log_head, trial_number, n_trials)
        trainer.fit(model, train_dataloader=train_dl, val_dataloaders=val_dl)

        # return the value for the metric specified in the start script
        if patience > 0:
            # return the best score while early stopping is applied
            val_error = early_stopping_callback.best_score.item()
        else:
            val_error = metrics_callback.metrics[-1][metric].item()

        logging.info('%s finished fitting for trial %s with %s = %s', log_head, trial_number, metric, val_error)

    if best_trial_retrain:
        ckpt_path = glob.glob(dirpath+'best_trial_retrain_model' + '*.ckpt')[0].replace('\\', '/')
        model.load_state_dict(torch.load(ckpt_path)['state_dict'])
        model.eval()

        index_dl = ['training set', 'validation set', 'test set']
        dataset_dl = [train_dl, val_dl, test_dl]
        results_metric = []
        for index_, dataset_ in zip(index_dl, dataset_dl):
            test_result = trainer.test(model, test_dataloaders=dataset_)[0]
            test_result['phase'] = index_
            results_metric.append(test_result)

            predictions = list(model.test_predictions)
            if not inverse:
                pred_df = pd.DataFrame(list(standard_score_transform(transformer, np.array(predictions[0]))), columns=['Measured PCE'])
                pred_df['Predicted PCE'] = list(standard_score_transform(transformer, np.array(predictions[1])))
            else:
                pred_df = pd.DataFrame(predictions[0], columns=['Measured PCE'])
                pred_df['Predicted PCE'] = predictions[1]
            pred_df.to_csv(dirpath+'predictions_{}.csv'.format(index_.replace(' ', '_')))

        pd.DataFrame(results_metric).to_csv(dirpath+'best_trial_retrain_model_result.csv')

        if regression_plot:
            oscml.visualization.util_sns_plot.prediction_plot(dirpath, dirpath + 'predictions_training_set.csv',
                                                              dirpath + 'predictions_validation_set.csv',
                                                              dirpath + 'predictions_test_set.csv')

    if epochs > 0:
        return val_error
    return None


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


def objective(trial, config, df_train, df_val, df_test, transformer, log_dir, total_number_trials,
              best_trial_retrain=False, z_transform_inverse_prediction=False, regression_plot=False):
    # TODO how does retraining work for RF and SVR?
    # release GPU memory before start each trial
    torch.cuda.empty_cache()

    # init parameters from config file
    model_name = config['model']['name']
    training_params = get_training_params(trial, config['training'])
    cv = config['training']['cross_validation']
    metric = training_params['metric']
    seed = config['numerical_settings']['seed']
    split = config['dataset']['split']
    trial_number = trial.number
    std = None

    # deal with RF and SVR models first
    if model_name == 'RF':
        model, x_train, y_train, x_val, y_val, x_test, y_test = oscml.hpo.hpo_rf.create(trial, config, df_train, df_val, df_test, training_params)
        metric_value = oscml.utils.util_sklearn.train_and_test(x_train, y_train, x_val, y_val, x_test, y_test, model,
                                                               training_params['cross_validation'],
                                                               training_params['metric'], log_dir,
                                                               seed=seed,
                                                               trial_number=trial_number,
                                                               best_trial_retrain=best_trial_retrain,
                                                               transformer=transformer,
                                                               inverse=z_transform_inverse_prediction,
                                                               regression_plot=regression_plot)
    elif model_name == 'SVR':
        model, x_train, y_train, x_val, y_val, x_test, y_test = oscml.hpo.hpo_svr.create(trial, config, df_train, df_val, df_test, training_params)
        metric_value = oscml.utils.util_sklearn.train_and_test(x_train, y_train, x_val, y_val, x_test, y_test, model,
                                                               training_params['cross_validation'],
                                                               training_params['metric'], log_dir,
                                                               seed=seed,
                                                               trial_number=trial_number,
                                                               best_trial_retrain=best_trial_retrain,
                                                               transformer=transformer,
                                                               inverse=z_transform_inverse_prediction,
                                                               regression_plot=regression_plot)

    # then move to BILSTM, AttentiveFP, and SimpleGNN models
    elif model_name == 'BILSTM' or model_name == 'AttentiveFP' or model_name == 'SimpleGNN':
        # apply cross-validation
        if isinstance(cv, int) and cv > 1:
            if not best_trial_retrain:
                kf = KFold(n_splits=cv, random_state=seed, shuffle=True)
                assert df_val is None, "validation set should be added to training set for cross validation"
                kf.get_n_splits(df_train)
                cv_index = 1
                cv_metric = []
                for train_index, val_index in kf.split(df_train):
                    logging.info('[trial %s] run %s of %s fold cross-validation', trial_number, cv_index, cv)
                    model, train_dl, val_dl, test_dl = get_model_and_data(model_name, trial, config,
                                                                          df_train.iloc[train_index], df_train.iloc[val_index],
                                                                          df_test, training_params, transformer, log_dir)
                    metric_value = fit_or_test(model, train_dl, val_dl, test_dl, training_params, log_dir,
                                               trial, trial_number, total_number_trials, str(cv_index))
                    cv_index += 1
                    cv_metric.append(metric_value)
                metric_value = np.array(cv_metric).mean()
                std = np.array(cv_metric).std()

                logging.info('[trial %s - finished %s fold cross-validation] %s: %s', trial_number,
                             training_params['cross_validation'],
                             training_params['metric'],
                             cv_metric)
                logging.info('[trial %s - finished %s fold cross-validation] %s mean: %s', trial_number,
                             training_params['cross_validation'],
                             training_params['metric'],
                             metric_value)
                logging.info('[trial %s - finished %s fold cross-validation] %s variance: %s', trial_number,
                             training_params['cross_validation'],
                             training_params['metric'],
                             np.array(cv_metric).var())
            else:
                rs = ShuffleSplit(n_splits=1, test_size=1 / cv, random_state=seed + 1)
                rs.get_n_splits(df_train)
                for train_index, val_index in rs.split(df_train):
                    model, train_dl, val_dl, test_dl, = get_model_and_data(model_name, trial, config,
                                                                           df_train.iloc[train_index],
                                                                           df_train.iloc[val_index],
                                                                           df_test, training_params, transformer,
                                                                           log_dir)
                    metric_value = fit_or_test(model=model, train_dl=train_dl, val_dl=val_dl, test_dl=test_dl,
                                               training_params=training_params, log_dir=log_dir,
                                               trial=trial, trial_number=trial_number, n_trials=total_number_trials,
                                               cv_index='', best_trial_retrain=best_trial_retrain,
                                               transformer=transformer, inverse=z_transform_inverse_prediction,
                                               regression_plot=regression_plot)

        # normal training and testing
        else:
            model, train_dl, val_dl, test_dl = get_model_and_data(model_name, trial, config, df_train, df_val, df_test,
                                                                  training_params, transformer, log_dir)
            metric_value = fit_or_test(model=model, train_dl=train_dl, val_dl=val_dl, test_dl=test_dl,
                                       training_params=training_params, log_dir=log_dir,
                                       trial=trial, trial_number=trial_number, n_trials=total_number_trials,
                                       cv_index='', best_trial_retrain=best_trial_retrain,
                                       transformer=transformer, inverse=z_transform_inverse_prediction,
                                       regression_plot=regression_plot)

    else:
        return None

    logging.info('objective value for trial %s with %s = %s, std=%s', trial_number, metric, metric_value, std)

    return metric_value
