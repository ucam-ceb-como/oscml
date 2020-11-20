import logging

import oscml.data.dataset
import oscml.models.model_gnn
from oscml.utils.util_config import set_config_param

def create(trial, config, df_train, df_val, df_test, optimizer, transformer):

    type_dict = config['model']['type_dict']
    batch_size = config['training']['batch_size']
    info = oscml.data.dataset.get_dataset_info(type_dict)
    node_type_number = len(info.node_types)

    #dataloaders
    train_dl, val_dl, test_dl = oscml.models.model_gnn.get_dataloaders(type_dict, df_train, df_val, df_test,
            transformer, batch_size=batch_size)

    # set model parameters from the config file
    #--------------------------------------
    model_params = {}
    for key, value in config['model']['model_specific'].items():
        model_params.update({key: set_config_param(trial=trial,param_name=key,param=value, all_params=model_params)})

    # add output dimension to the mlp_units
    model_params['mlp_units'] = model_params.get('mlp_units', []) + [1]

    # add extra params not defined in the config file
    extra_params =  {
        # additional non-hyperparameter values
        'node_type_number': node_type_number, #len(oscml.data.dataset_hopv15.ATOM_TYPES_HOPV15),
        'padding_index': 0,
        'target_mean': transformer.target_mean,
        'target_std': transformer.target_std,
    }
    model_params.update(extra_params)
    logging.info('model params=%s', model_params)

    model_params.pop('mlp_layers',None) # this is not needed for the model creation
    model_params.pop('conv_layers',None) # this is not needed for the model creation
    model = oscml.models.model_gnn.SimpleGNN(**model_params, optimizer=optimizer)

    return model, train_dl, val_dl, test_dl
