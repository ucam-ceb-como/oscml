import logging

import oscml.data.dataset
import oscml.models.model_gnn
from oscml.hpo.optunawrapper import set_config_param
def create(trial, config, df_train, df_val, df_test, optimizer, transformer, dataset):

    info = oscml.data.dataset.get_dataset_info(dataset)
    node_type_number = len(info.node_types)

    #dataloaders
    train_dl, val_dl, test_dl = oscml.models.model_gnn.get_dataloaders(dataset, df_train, df_val, df_test,
            transformer, batch_size=250)

    # define model and params
    model_specific = config['model']['model_specific'].copy()
    conv_dims_direction = config['model']['model_specific']['conv_dims'].get('direction')
    mlp_units_direction = config['model']['model_specific']['mlp_units'].get('direction')

    embedding_dim = set_config_param(trial=trial,param_name='embedding_dim',param=model_specific['embedding_dimension']) #trial.suggest_int('embedding_dim', 8, 256)
    conv_dims = []
    conv_layers =  set_config_param(trial=trial,param_name='embedding_dim',param=model_specific['conv_layers']) #trial.suggest_int('conv_layers', 1, 4)
    suggested_units = embedding_dim
    for l in range(conv_layers):
        if conv_dims_direction=="decreasing":
            model_specific['conv_dims']['high'] = suggested_units
        elif conv_dims_direction=="increasing":
            model_specific['conv_dims']['low'] = suggested_units
        elif conv_dims_direction=="constant":
            model_specific['conv_dims'] = suggested_units
        suggested_units = set_config_param(trial=trial,param_name='conv_dims_{}'.format(l),param=model_specific['conv_dims']) #trial.suggest_int('conv_dims_{}'.format(l), 10, max_units)
        conv_dims.append(suggested_units)
       # max_units = suggested_units

    mlp_layers = set_config_param(trial=trial,param_name='mlp_layers',param=model_specific['mlp_layers'])  #trial.suggest_int('mlp_layers', 1, 4)
    # the number of units of the last gnn layer is the input dimension for the mlp
    mlp_units = []
    mlp_dropout_rate = set_config_param(trial=trial,param_name='mlp_dropout',param=model_specific['mlp_dropout'])  #trial.suggest_float('mlp_dropout', 0.1, 0.3)
    mlp_dropouts = []
    for l in range(mlp_layers):
        suggested_units = set_config_param(trial=trial,param_name='mlp_units_{}'.format(l),param=model_specific['mlp_units']) #trial.suggest_int('mlp_units_{}'.format(l), 5, max_units)
        mlp_units.append(suggested_units)
        if mlp_units_direction=="decreasing":
            model_specific['mlp_units']['high'] = suggested_units
        elif mlp_units_direction=="increasing":
            model_specific['mlp_units']['low'] = suggested_units
        elif mlp_units_direction=="constant":
            model_specific['mlp_units'] = suggested_units
        mlp_dropouts.append(mlp_dropout_rate)

    # add output dimension
    mlp_units.append(1)

    model_params =  {
        'embedding_dim': embedding_dim,
        'conv_dims': conv_dims,
        'mlp_units': mlp_units,
        'mlp_dropouts': mlp_dropouts,
        # additional non-hyperparameter values
        'node_type_number': node_type_number, #len(oscml.data.dataset_hopv15.ATOM_TYPES_HOPV15),
        'padding_index': 0,
        'target_mean': transformer.target_mean,
        'target_std': transformer.target_std,
    }

    logging.info('model params=%s', model_params)

    model = oscml.models.model_gnn.SimpleGNN(**model_params, optimizer=optimizer)

    return model, train_dl, val_dl, test_dl
