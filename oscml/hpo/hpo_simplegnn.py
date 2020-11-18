import logging

import oscml.data.dataset
import oscml.models.model_gnn

def create(trial, config, df_train, df_val, df_test, optimizer, transformer, dataset):

    info = oscml.data.dataset.get_dataset_info(dataset)
    node_type_number = len(info.node_types)

    #dataloaders
    train_dl, val_dl, test_dl = oscml.models.model_gnn.get_dataloaders(dataset, df_train, df_val, df_test, 
            transformer, batch_size=250)
    
    # define model and params   
    embedding_dim = trial.suggest_int('embedding_dim', 8, 256)
    conv_dims = []
    conv_layers =  trial.suggest_int('conv_layers', 1, 4)
    max_units = 256
    for l in range(conv_layers):
        suggested_units = trial.suggest_int('conv_dims_{}'.format(l), 10, max_units)
        conv_dims.append(suggested_units)
        max_units = suggested_units

    mlp_layers =  trial.suggest_int('mlp_layers', 1, 4)
    # the number of units of the last gnn layer is the input dimension for the mlp
    mlp_units = []
    mlp_dropout_rate = trial.suggest_float('mlp_dropout', 0.1, 0.3)
    mlp_dropouts = []
    for l in range(mlp_layers):
        suggested_units = trial.suggest_int('mlp_units_{}'.format(l), 5, max_units)
        mlp_units.append(suggested_units)
        max_units = suggested_units
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

    model = oscml.models.model_gnn.GNNSimple(**model_params, optimizer=optimizer)

    return model, train_dl, val_dl, test_dl
