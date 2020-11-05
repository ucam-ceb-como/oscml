import argparse
import datetime
import json
import logging
import os

import numpy as np
import optuna
import optuna.samplers
import pytorch_lightning as pl
import torch

import oscml
import oscml.utils.util
from oscml.utils.util import concat
import oscml.utils.util_lightning


class MetricsCallback(pl.Callback):

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


def get_attrs(trial):
    if isinstance(trial, optuna.trial.FixedTrial):
        user_attrs = trial.user_attrs
    else:
        user_attrs = trial.study.user_attrs
    
    init_attrs = None
    if 'init_attrs' in user_attrs:
        init_attrs = user_attrs['init_attrs']
    
    return user_attrs, init_attrs


def fit(model_instance, train_dl, val_dl, trainer_params, trial):

    user_attrs, _ = get_attrs(trial)
    src = user_attrs['src']
    dst = user_attrs['dst']
    epochs = user_attrs['epochs']
    n_trials = user_attrs['trials']
    metric = user_attrs['metric']
    log_dir = user_attrs['log_dir']

    trial_number = trial.number
    logging.info(concat('fitting trial ', trial_number, ' / ', n_trials))

    # create standard params for Ligthning trainer
    trainer_params = oscml.utils.util_lightning.get_standard_params_for_trainer(metric)

    # create Lightning metric logger that logs metric values for each trial in its own csv file
    # version='' means that no version-subdirectory is created
    csv_logger = pl.loggers.CSVLogger(save_dir=log_dir, 
                                      name='trial_' + str(trial_number), 
                                      version='')     

    # create callbacks for Optuna for receiving the metric values from Lightning and for
    # pruning trials
    metrics_callback = MetricsCallback()
    pruning_callback = optuna.integration.PyTorchLightningPruningCallback(trial, monitor=metric)
    callbacks = [metrics_callback, pruning_callback]
  
    # put all trainer params together
    trainer_params.update({
        'max_epochs': epochs,
        'logger': csv_logger,
        'callbacks': callbacks
    })

    logging.info(concat('params for Lightning trainer=', trainer_params))
    
    # start fitting
    trainer = pl.Trainer(**trainer_params)
    trainer.fit(model_instance, train_dataloader=train_dl, val_dataloaders=val_dl)

    # return the value for the metric specified in the start script
    value =  metrics_callback.metrics[-1][metric].item()
    logging.info(concat('finished fitting for trial ', trial_number, ' with ', metric, '=', value))
    return value


def create_objective_decorator(objective, n_trials):
        def decorator(trial):
            logging.info(concat('starting trial ', trial.number, ' / ', n_trials))
            value = objective(trial)
            logging.info(concat('finished trial ', trial.number, ' / ', n_trials))
            return value

        return decorator


def create_study(direction, seed):
    # pruner = optuna.pruners.MedianPruner()
    # pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)
    # pruner = optuna.pruners.PercentilePruner(25.0, n_startup_trials=5, n_warmup_steps=30, interval_steps=10) #keep top 25%
    # #pruner = ThresholdPruner(upper=1.0)
    #pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=epochs, reduction_factor=3)
    pruner = None
    sampler = optuna.samplers.TPESampler(consider_prior=True, n_startup_trials=10, seed=seed)
    study = optuna.create_study(direction=direction, pruner=pruner, sampler=sampler)
    return study


def start_hpo(init, objective, metric, direction, fixed_trial=None, seed=200, timeout_in_seconds=600, post_hpo=None):

    print('current working directory=', os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default='.')
    parser.add_argument("--dst", type=str, default='.')
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()
    args = parser.parse_args()

    # init file logging
    log_config_file = args.src + '/conf/logging.yaml'
    date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = args.dst + '/logs/hpo_' + date
    oscml.utils.util.init_file_logging(log_config_file, log_dir + '/oscml.log')

    logging.info('current working directory=' + os.getcwd())

    user_attrs = vars(args).copy()
    user_attrs.update({
        'metric': metric,
        'log_dir': log_dir
    })
    logging.info(user_attrs)
    logging.info({'direction': direction, 'seed': seed})

    try:
        np.random.seed(seed)
        torch.manual_seed(seed)

        if init:
            init_attrs = init(user_attrs)
            user_attrs['init_attrs'] = init_attrs
            logging.info('init finished with attributes=' + str(init_attrs))

        if fixed_trial:
            trial = optuna.trial.FixedTrial(params=fixed_trial)
            for key, value in user_attrs.items():
                trial.set_user_attr(key, value)
            logging.info('calling objective function with fixed trial')
            value = objective(trial)
            logging.info(concat('finished objective function call with ', metric, '=', value))
        else:
            study = create_study(direction, seed)
            for key, value in user_attrs.items():
                study.set_user_attr(key, value)

            if args.config:
                with open(args.config) as config_json:
                    #optuna_config = json.load(config_json)['HPO']['optuna']
                    config_model = json.load(config_json)[args.model]
                objective_fct = objective(config_model)
            else:
                objective_fct = objective

            decorator = create_objective_decorator(objective_fct, args.trials)

            logging.info('starting HPO')
            study.optimize(decorator, n_trials=args.trials, n_jobs=args.jobs,
                    timeout=timeout_in_seconds, gc_after_trial=True)
            path = log_dir + '/hpo_result.csv'
            log_and_save(study, path)

    except BaseException as exc:
        print(exc)
        logging.exception('finished with exception', exc_info=True)
        raise exc
    else:
        logging.info('finished successfully')


def log_and_save(study, path):
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    message = 'Study statistics: number of finished / pruned / complete trials='
    logging.info(concat(message, len(study.trials), len(pruned_trials), len(complete_trials)))

    trial = study.best_trial
    logging.info('Best trial number =' + str(trial.number))
    logging.info('Best trial value =' + str(trial.value))
    logging.info('Best trial params=' + str(trial.params))
  
    logging.info('Saving HPO results to ' + path)
    df = study.trials_dataframe()
    df.to_csv(path)