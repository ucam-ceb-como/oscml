import logging

import optuna.trial
import pytorch_lightning as pl

import oscml.data.dataset
import oscml.data.dataset_cep
import oscml.data.dataset_hopv15
import oscml.hpo.hpo_attentivefp
import oscml.hpo.hpo_bilstm
import oscml.hpo.hpo_simplegnn
import oscml.hpo.hpo_rf
import oscml.hpo.optunawrapper
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from oscml.utils.util_config import set_config_param
import oscml.utils.util_sklearn

class MetricsCallback(pl.Callback):

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)

def fit_or_test(model, train_dl, val_dl, test_dl, training_params,
                epochs, metric, log_dir, trial=None, trial_number=-1, n_trials=0):

    # create callbacks for Optuna for receiving the metric values from Lightning and for
    # pruning trials
    metrics_callback = MetricsCallback()
    callbacks = [metrics_callback]
    if trial:
        pruning_callback = optuna.integration.PyTorchLightningPruningCallback(trial, monitor=metric)
        callbacks.append(pruning_callback)

    early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=3, verbose=False, mode='min')
    callbacks.append(early_stopping_callback)

    logging.info('model for trial %s=%s', trial_number, model)

    # create standard params for Ligthning trainer

    # if the number of trials is 1 then save checkpoints for the last and best epoch
    # otherwise if HPO is running (i.e. unspecified time-contrained number of trials or finite number > 1 )
    # then save no checkpoints
    save_checkpoints =  (n_trials is not None and n_trials == 1)
    trainer_params = oscml.utils.util_lightning.get_standard_params_for_trainer(metric, save_checkpoints)

    # create Lightning metric logger that logs metric values for each trial in its own csv file
    # version='' means that no version-subdirectory is created
    csv_logger = pl.loggers.CSVLogger(save_dir=log_dir,
                                      name='trial_' + str(trial_number),
                                      version='')

    # put all trainer params together
    trainer_params.update({
        'max_epochs': training_params['epochs'],
        'logger': csv_logger,
        'callbacks': callbacks
    })

    logging.info('params for Lightning trainer=%s', trainer_params)

    trainer = pl.Trainer(**trainer_params)

    if epochs > 0:
        logging.info('fitting trial %s / %s', trial_number, n_trials)
        trainer.fit(model, train_dataloader=train_dl, val_dataloaders=val_dl)

        # return the value for the metric specified in the start script
        val_error =  metrics_callback.metrics[-1][metric].item()
        logging.info('finished fitting for trial %s with %s = %s', trial_number, metric, val_error)

    if test_dl:
        logging.info('testing trial %s / %s', trial_number, n_trials)
        test_result = trainer.test(model, test_dataloaders=test_dl)[0]
        #logging.info('result=%s', test_result)

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

def objective(trial, config, args, df_train, df_val, df_test, transformer):

    # init model and data loaders
    model_name = config['model']['name']

    if model_name == 'BILSTM':
        training_params = get_training_params(trial, config['training'])
        model, train_dl, val_dl, test_dl = oscml.hpo.hpo_bilstm.create(trial, config, df_train, df_val, df_test, training_params['optimiser'], transformer, args.dataset)
        tainer_type = "pl_lightning"

    elif model_name == 'AttentiveFP':
        training_params = get_training_params(trial, config['training'])
        model, train_dl, val_dl, test_dl = oscml.hpo.hpo_attentivefp.create(trial, config, args, df_train, df_val, df_test, training_params['optimiser'])
        tainer_type = "pl_lightning"

    elif model_name == 'SimpleGNN':
        training_params = get_training_params(trial, config['training'])
        model, train_dl, val_dl, test_dl = oscml.hpo.hpo_simplegnn.create(trial, config, df_train, df_val, df_test, training_params['optimiser'], transformer, args.dataset)
        tainer_type = "pl_lightning"

    elif model_name == 'RF':
        training_params = get_training_params(trial, config['training'])
        model, x_train, y_train, x_val, y_val = oscml.hpo.hpo_rf.create(trial, config, df_train, df_val, df_test, training_params, args.dataset)
        tainer_type = "scikit_learn"

    # fit on training set and calculate metric on validation set
    if tainer_type == "pl_lightning":
    trial_number = trial.number
        metric_value = fit_or_test(model, train_dl, val_dl, test_dl, training_params,
                                args.epochs, args.metric, args.log_dir, trial, trial_number, args.trials)
    else:
        metric_value = oscml.utils.util_sklearn.train_and_test(x_train, y_train, x_val, y_val, model,
                                                                  training_params['cross_validation'], training_params['criterion'])

    return metric_value
