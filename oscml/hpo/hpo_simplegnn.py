import logging

import oscml.data.dataset
import oscml.models.model_gnn
from oscml.utils.util_config import set_config_param, set_config_param_list
def create(trial, config, df_train, df_val, df_test, optimizer, transformer, dataset):

    info = oscml.data.dataset.get_dataset_info(dataset)
    node_type_number = len(info.node_types)

    #dataloaders
    train_dl, val_dl, test_dl = oscml.models.model_gnn.get_dataloaders(dataset, df_train, df_val, df_test,
            transformer, batch_size=250)

    # define model and params
    # copy model params to another dictionary, as we may need to modify some of its values
    model_specific = config['model']['model_specific'].copy()

    # start setting model parameters
    #--------------------------------------
    embedding_dim = set_config_param(trial=trial,param_name='embedding_dim',param=model_specific['embedding_dimension'])
    conv_layers =  set_config_param(trial=trial,param_name='conv_layers',param=model_specific['conv_layers'])
    conv_dims = set_config_param_list(trial=trial,param_name='conv_dims',param=model_specific['conv_dims'],length=conv_layers)
    mlp_layers = set_config_param(trial=trial,param_name='mlp_layers',param=model_specific['mlp_layers'])
    # the number of units of the last gnn layer is the input dimension for the mlp
    mlp_units = set_config_param_list(trial=trial,param_name='mlp_units',param=model_specific['mlp_units'],length=mlp_layers)
    mlp_dropouts = set_config_param_list(trial=trial,param_name='mlp_dropouts',param=model_specific['mlp_dropouts'],length=mlp_layers)

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
