import logging

import oscml.hpo.optunawrapper
import oscml.models.model_example_mlp_mnist

def objective(trial):

    # define data loaders and params
    data_loader_params = {
        'mnist_dir': './tmp', 
        'batch_size': trial.suggest_int('batch_size', 16, 256)
    }
    train_dl, val_dl = oscml.models.model_example_mlp_mnist.get_mnist(**data_loader_params)

    # define model and params
    layers =  trial.suggest_int('layers', 1, 4)
    units = []
    dropouts = []
    max_units = 100
    for l in range(layers):
        postfix = '_l' + str(l)
        suggested_units = trial.suggest_int('units'+postfix, 10, max_units)
        units.append(suggested_units)
        max_units = suggested_units
        dropouts.append(trial.suggest_float('dropouts'+postfix, 0.1, 0.3))

    model_params =  {
        'number_classes': 10, 
        'layers': layers, 
        'units': units, 
        'dropouts': dropouts,
        'optimizer':  trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD']), 
        'optimizer_lr': trial.suggest_float('optimizer_lr', 1e-5, 1e-1, log=True)
    }

    model_instance = oscml.models.model_example_mlp_mnist.MlpWithLightning(**model_params)
    
    # fit on training set and calculate metric on validation set
    trainer_params = {
    }
    metric_value =  oscml.hpo.optunawrapper.fit(model_instance, train_dl, val_dl, trainer_params, trial)
    return metric_value


def fixed_trial():
    return {
        'number_classes': 10, 
        'layers': 2,
        'units_l0': 20,
        'units_l1': 10, 
        'dropouts_l0': 0.2,
        'dropouts_l1': 0.15,
        'optimizer': 'Adam',
        'optimizer_lr': 0.001,
        'batch_size': 128
    }
   

def start():
    return oscml.hpo.optunawrapper.start_hpo(init=None, objective=objective, metric='val_acc', direction='maximize',
        fixed_trial_params=fixed_trial())

if __name__ == '__main__':
    start()