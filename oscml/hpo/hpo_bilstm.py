import logging

import oscml.data.dataset
import oscml.models.model_bilstm
from oscml.utils.util_config import set_config_param, set_config_param_list

def create(trial, config, df_train, df_val, df_test, optimizer, transformer, dataset):

    info = oscml.data.dataset.get_dataset_info(dataset)
    number_subgraphs = info.number_subgraphs()
    max_sequence_length = info.max_sequence_length

    # dataloaders
    train_dl, val_dl, test_dl = oscml.models.model_bilstm.get_dataloaders(dataset, df_train, df_val, df_test,
            transformer, batch_size=250, max_sequence_length=max_sequence_length)

    # define models and params
    model_specific = config['model']['model_specific'].copy()

    embedding_dim = set_config_param(trial=trial,param_name='embedding_dim',param=model_specific['embedding_dimension'])
    lstm_hidden_dim = embedding_dim
    mlp_layers = set_config_param(trial=trial,param_name='mlp_layers',param=model_specific['mlp_layers'])
    mlp_units = set_config_param_list(trial=trial,param_name='mlp_units',param=model_specific['mlp_units'],length=mlp_layers)
    mlp_dropouts = set_config_param_list(trial=trial,param_name='mlp_dropouts',param=model_specific['mlp_dropouts'],length=mlp_layers)

    # add output dimension
    mlp_units.append(1)

    model_params =  {
        'embedding_dim': embedding_dim,
        'lstm_hidden_dim': lstm_hidden_dim,
        'mlp_units': mlp_units,
        'mlp_dropouts': mlp_dropouts,
        # additional non-hyperparameter values
        'number_of_subgraphs': number_subgraphs,
        'padding_index': 0,
        'target_mean': transformer.target_mean,
        'target_std': transformer.target_std,
    }

    logging.info('model params=%s', model_params)

    model = oscml.models.model_bilstm.BiLstmForPce(**model_params, optimizer=optimizer)

    return model, train_dl, val_dl, test_dl
