import logging

import oscml.data.dataset
import oscml.models.model_bilstm

def create(trial, config, df_train, df_val, df_test, optimizer, transformer, dataset):
    
    info = oscml.data.dataset.get_dataset_info(dataset)
    number_subgraphs = info.number_subgraphs()
    max_sequence_length = info.max_sequence_length

    # dataloaders
    train_dl, val_dl, test_dl = oscml.models.model_bilstm.get_dataloaders(dataset, df_train, df_val, df_test,
            transformer, batch_size=250, max_sequence_length=max_sequence_length)

    # define models and params
    subgraph_embedding_dim = trial.suggest_int('subgraph_embedding_dim', 8, 256)
    lstm_hidden_dim = subgraph_embedding_dim
    mlp_layers =  trial.suggest_int('mlp_layers', 1, 4)
    mlp_units = []
    mlp_dropout_rate = trial.suggest_float('mlp_dropout', 0.01, 0.2)
    mlp_dropouts = []
    #max_units = 2 * lstm_hidden_dim
    for l in range(mlp_layers):
        suggested_units = trial.suggest_categorical('mlp_units_{}'.format(l), [20, 40, 60, 80, 120, 160, 200])
        mlp_units.append(suggested_units)
        #max_units = suggested_units
        mlp_dropouts.append(mlp_dropout_rate)

    # add output dimension
    mlp_units.append(1)

    model_params =  {
        'subgraph_embedding_dim': subgraph_embedding_dim,
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
