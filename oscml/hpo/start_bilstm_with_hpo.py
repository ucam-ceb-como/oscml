import logging
import types

import oscml.data.dataset
import oscml.data.dataset_cep
import oscml.hpo.optunawrapper
import oscml.models.model_bilstm


def init(user_attrs):

    # read data and preprocess, e.g. standarization, splitting into train, validation and test set
    src = user_attrs['src']
    path = oscml.data.dataset.path_cepdb_25000(src)
    df_train, df_val, df_test = oscml.data.dataset.read_and_split(path)
    # just for development: use 10% of the 25000 samples
    #df_train = df_train[:1500].copy()
    #df_val = df_val[:500].copy()
    #df_test = df_test[:500].copy()
    transformer = oscml.data.dataset.create_transformer(df_train, column_target='pce', column_x='SMILES_str')
    
    return (df_train, df_val, df_test, transformer)


def objective(trial):

    _, init_attrs = oscml.hpo.optunawrapper.get_attrs(trial)
    df_train = init_attrs[0]
    df_val = init_attrs[1]
    df_test = init_attrs[2]
    transformer = init_attrs[3]

    # define data loader and params

    data_loader_params = {
        'train': df_train,
        'val': df_val,
        'test': None, 
        'batch_size': 250, 
        'max_sequence_length': 60,
        'smiles_fct': transformer.transform_x,
        'target_fct': transformer.transform, 
    }

    train_dl, val_dl, test_dl = oscml.models.model_bilstm.get_dataloaders_CEP(**data_loader_params)


    # define models and params
    subgraph_embedding_dim = trial.suggest_int('subgraph_embedding_dim', 8, 256)
    lstm_hidden_dim = subgraph_embedding_dim
    mlp_layers =  trial.suggest_int('mlp_layers', 1, 4)
    mlp_units = []
    mlp_dropout_rate = trial.suggest_float('mlp_dropout', 0.05, 0.3)
    mlp_dropouts = []
    max_units = 2 * lstm_hidden_dim
    for l in range(mlp_layers):
        suggested_units = trial.suggest_int('mlp_units_{}'.format(l), 5, max_units)
        mlp_units.append(suggested_units)
        max_units = suggested_units
        mlp_dropouts.append(mlp_dropout_rate)

    # add output dimension 
    mlp_units.append(1)

    model_params =  {
        'subgraph_embedding_dim': subgraph_embedding_dim,
        'lstm_hidden_dim': lstm_hidden_dim,
        'mlp_units': mlp_units,
        'mlp_dropouts': mlp_dropouts,
        'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD']), 
        'optimizer_lr': trial.suggest_float('optimizer_lr', 1e-5, 1e-1, log=True),
        # additional non-hyperparameter values
        'number_of_subgraphs': 60,
        'padding_index': 0,
        'target_mean': transformer.target_mean, 
        'target_std': transformer.target_std,
    }

    logging.info('model params=' + str(model_params))

    model_instance = oscml.models.model_bilstm.BiLstmForPce(**model_params)

    
    # fit on training set and calculate metric on validation set
    trainer_params = {
    }
    metric_value =  oscml.hpo.optunawrapper.fit(model_instance, train_dl, val_dl, trainer_params, trial)
    return metric_value


def fixed_trial():
    return {
        'subgraph_embedding_dim': 128,
        #'lstm_hidden_dim': 128,  # currently set to subgraph_embedding_dim
        'mlp_layers': 3,
        'mlp_units_0': 64,
        'mlp_units_1': 32,
        'mlp_units_2': 32,       
        'mlp_dropout': 0.1,
        'optimizer': 'Adam', 
        'optimizer_lr': 0.001,
        #'batch_size': 250
    }


def start():
    return oscml.hpo.optunawrapper.start_hpo(init=init, objective=objective, metric='val_loss', direction='minimize',
        fixed_trial_params=fixed_trial())

if __name__ == '__main__':
    start()