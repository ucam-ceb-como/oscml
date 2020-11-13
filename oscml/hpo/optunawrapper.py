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

os.environ["SLURM_JOB_NAME"]="bash"




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

    return fit_or_test(model_instance, train_dl, val_dl, None, trainer_params, 
                epochs, metric, log_dir, trial, trial_number, n_trials)


def fit_or_test(model, train_dl, val_dl, test_dl, trainer_params, 
                epochs, metric, log_dir, trial=None, trial_number=-1, n_trials=0):

    # create callbacks for Optuna for receiving the metric values from Lightning and for
    # pruning trials
    metrics_callback = MetricsCallback()
    callbacks = [metrics_callback] 
    if trial:
        pruning_callback = optuna.integration.PyTorchLightningPruningCallback(trial, monitor=metric)
        callbacks.append(pruning_callback)
        
    logging.info(concat('model for trial', trial_number, '=', model))
 

    # create standard params for Ligthning trainer
    trainer_params = oscml.utils.util_lightning.get_standard_params_for_trainer(metric)

    # create Lightning metric logger that logs metric values for each trial in its own csv file
    # version='' means that no version-subdirectory is created
    csv_logger = pl.loggers.CSVLogger(save_dir=log_dir, 
                                      name='trial_' + str(trial_number), 
                                      version='')     

    # put all trainer params together
    trainer_params.update({
        'max_epochs': epochs,
        'logger': csv_logger,
        'callbacks': callbacks
    })

    logging.info(concat('params for Lightning trainer=', trainer_params))
    
    trainer = pl.Trainer(**trainer_params)

    if epochs > 0:
        logging.info(concat('fitting trial ', trial_number, ' / ', n_trials))
        trainer.fit(model, train_dataloader=train_dl, val_dataloaders=val_dl)

        # return the value for the metric specified in the start script
        value =  metrics_callback.metrics[-1][metric].item()
        logging.info(concat('finished fitting for trial ', trial_number, ' with ', metric, '=', value))
        return value

    else:
        logging.info(concat('testing trial ', trial_number, ' / ', n_trials))
        result = trainer.test(model, test_dataloaders=test_dl)
        logging.info('result=' + str(result[0]))
        return result[0]

def create_objective_decorator(objective, n_trials):
        def decorator(trial):
            try:
                logging.info(concat('starting trial ', trial.number, ' / ', n_trials))
                value = objective(trial)
                logging.info(concat('finished trial ', trial.number, ' / ', n_trials))
                return value
            except optuna.exceptions.TrialPruned as exc:
                message = 'pruned trial, trial number=' + str(trial.number)
                logging.info(message)
                raise exc
            except Exception as exc:
                message = 'failed trial, trial number=' + str(trial.number)
                logging.exception(message, exc_info=True)
                raise exc

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

def get_statistics(study):
    running_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    return {'all': len(study.trials), 'running': len(running_trials), 'completed': len(completed_trials), 
            'pruned': len(pruned_trials), 'failed': len(failed_trials)}

def callback_on_trial_finished(study, trial):
    statistics = get_statistics(study)
    logging.info(concat('current study statistics: number of trials=', statistics))
    if statistics['failed'] >= 50:
        logging.error('THE MAXIMUM NUMBER OF FAILED TRIALS HAS BEEN REACHED, AND THE STUDY WILL STOP NOW.')
        study.stop()


def start_hpo(init, objective, metric, direction, fixed_trial_params=None, seed=200, resume=None, post_hpo=None):

    print('current working directory=', os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='.')
    parser.add_argument('--dst', type=str, default='.')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--trials', type=int, default=None)
    parser.add_argument('--timeout', type=int, default=None, help='Stop study after the given number of second(s). If this argument is not set, the study is executed without time limitation.') 
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--fixedtrial', type=bool, default=False)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    # init file logging
    log_config_file = args.src + '/conf/logging.yaml'
    date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = args.dst + '/logs/hpo_' + date
    oscml.utils.util.init_file_logging(log_config_file, log_dir + '/oscml.log')

    logging.info('current working directory=' + os.getcwd())

    #optuna.logging.enable_default_handler()
    #optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    #optuna.logging.disable_default_handler() 
    #optuna.logging.set_verbosity(optuna.logging.DEBUG)

    if args.seed:
        seed = args.seed

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
            logging.info('init finished')

        if args.fixedtrial:
            assert args.trials == None or args.trials == 1
            trial = optuna.trial.FixedTrial(fixed_trial_params)
            for key, value in user_attrs.items():
                trial.set_user_attr(key, value)
            logging.info('calling objective function with fixed trial')
            best_value = objective(trial)
            logging.info(concat('finished objective function call with ', metric, '=', best_value))
        elif args.ckpt:
            resume_attrs = {
                'ckpt': args.ckpt, 
                'src': args.src, 
                'log_dir': log_dir, 
                'dataset': args.dataset,
                'epochs': args.epochs, 
                'metric': metric
            }
            best_value = resume(**resume_attrs)
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
            study.optimize(decorator, n_trials=args.trials, n_jobs=args.jobs, timeout=args.timeout, 
                    catch = (RuntimeError, ValueError, TypeError), callbacks=[callback_on_trial_finished],
                    gc_after_trial=True)
            logging.info('finished HPO')
            path = log_dir + '/hpo_result.csv'
            log_and_save(study, path)
            best_value = study.best_trial.value

        return best_value

    except BaseException as exc:
        print(exc)
        logging.exception('finished with exception', exc_info=True)
        raise exc
    else:
        logging.info('finished successfully')


def log_and_save(study, path):

    logging.info('Saving HPO results to ' + path)
      
    df = study.trials_dataframe()
    df.to_csv(path)

    logging.info(concat('final study statistics: number of trials=', get_statistics(study)))

    trial = study.best_trial
    logging.info('best trial number =' + str(trial.number))
    logging.info('best trial value =' + str(trial.value))
    logging.info('best trial params=' + str(trial.params))