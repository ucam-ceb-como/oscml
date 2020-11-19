import logging

import optuna.trial
import pytorch_lightning as pl

import oscml.data.dataset
import oscml.data.dataset_cep
import oscml.data.dataset_hopv15
#import oscml.hpo.hpo_attentivefp
import oscml.hpo.hpo_bilstm
import oscml.hpo.hpo_simplegnn
import oscml.hpo.optunawrapper


def get_attrs(trial):
    if isinstance(trial, optuna.trial.FixedTrial):
        user_attrs = trial.user_attrs
    else:
        user_attrs = trial.study.user_attrs

    #init_attrs = None
    #if 'init_attrs' in user_attrs:
    #    init_attrs = user_attrs['init_attrs']

    return user_attrs

class MetricsCallback(pl.Callback):

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)

def fit(model_instance, train_dl, val_dl, trainer_params, trial):

    user_attrs = get_attrs(trial)
    #src = user_attrs['src']
    #dst = user_attrs['dst']
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

    logging.info('model for trial %s=%s', trial_number, model)

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

    logging.info('params for Lightning trainer=%s', trainer_params)

    trainer = pl.Trainer(**trainer_params)

    if epochs > 0:
        logging.info('fitting trial %s / %s', trial_number, n_trials)
        trainer.fit(model, train_dataloader=train_dl, val_dataloaders=val_dl)

        # return the value for the metric specified in the start script
        value =  metrics_callback.metrics[-1][metric].item()
        logging.info('finished fitting for trial %s with %s = %s', trial_number, metric, value)
        return value

    else:
        logging.info('testing trial %s / %s', trial_number, n_trials)
        result = trainer.test(model, test_dataloaders=test_dl)
        logging.info('result=%s', result[0])
        return result[0]

def get_optimizer_params(trial):
    name =  trial.suggest_categorical('opt_name', ['Adam', 'RMSprop', 'SGD'])
    optimizer = {
        'name': name,
        'lr': trial.suggest_loguniform('lr', 1e-5, 1e-1),
        'weight_decay': trial.suggest_uniform('weight_decay', 0, 0.01)
    }
    if name in ['RMSprop', 'SGD']:
        optimizer['momentum'] = trial.suggest_uniform('momentum', 0, 0.01)
    if name == 'SGD':
        optimizer['nesterov'] = trial.suggest_categorical('nesterov', [True, False])
    return optimizer

def objective(trial, config, df_train, df_val, df_test, transformer, log_dir, featurizer):

    user_attrs = get_attrs(trial)
    #model_id = user_attrs['model']
    model_name = config['model']['name']

    # read data and preprocess, e.g. standarization, splitting into train, validation and test set
    #src = user_attrs['src']
    dataset = user_attrs['dataset']
    #df_train, df_val, df_test, transformer = oscml.data.dataset.get_dataframes(dataset=dataset, src=src, train_size=283, test_size=30)

    # init model and data loaders
    if model_name == 'BILSTM':
        optimizer = get_optimizer_params(trial)
        model, train_dl, val_dl, test_dl = oscml.hpo.hpo_bilstm.create(trial, config, df_train, df_val, df_test, optimizer, transformer, dataset)

    #elif model_name == 'AttentiveFP':
    #    optimizer = get_optimizer_params(trial)
    #    model, train_dl, val_dl, test_dl = oscml.hpo.hpo_attentivefp.create(trial, config, df_train, df_val, df_test, optimizer)

    elif model_name == 'SimpleGNN':
        optimizer = get_optimizer_params(trial)
        model, train_dl, val_dl, test_dl = oscml.hpo.hpo_simplegnn.create(trial, config, df_train, df_val, df_test, optimizer, transformer, dataset)

    # fit on training set and calculate metric on validation set
    trainer_params = {}
    metric_value = fit(model, train_dl, val_dl, trainer_params, trial)
    return metric_value
