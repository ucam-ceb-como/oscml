import logging
import types

import oscml.start
import oscml.data.dataset
import oscml.data.dataset_hopv15
import oscml.hpo.optunawrapper
import oscml.models.model_gnn


def init(user_attrs):

    # read data and preprocess, e.g. standarization, splitting into train, validation and test set
    src_dir = user_attrs['src']
    path = oscml.start.path_hopv_15(src_dir)
    df = oscml.data.dataset_hopv15.read(path)
    df = oscml.data.dataset.clean_data(df, None, 'smiles', 'pce')

    df_train, df_val, df_test, transformer = oscml.data.dataset.split_data_frames_and_transform(
            df, column_smiles='smiles', column_target='pce', train_size=283, test_size=30)
    
    return (df_train, df_val, df_test, transformer)


def objective(trial):

    _, init_attrs = oscml.hpo.optunawrapper.get_attrs(trial)
    df_train = init_attrs[0]
    df_val = init_attrs[1]
    transformer = init_attrs[3]

    # define data loader and params
    node2index = oscml.data.dataset_hopv15.ATOM_TYPES_HOPV15
    mol2seq = oscml.models.model_gnn.Mol2seq_simple(node2index, fix=True, oov=True)

    data_loader_params = {
        'train':df_train,
        'val': df_val,
        'test': None, 
        'transformer': transformer,
        'batch_size': 20, 
        'mol2seq': mol2seq
    }

    train_dl, val_dl = oscml.models.model_gnn.get_dataloaders(**data_loader_params)


    # define model and params   
    gnn_layers =  trial.suggest_int('gnn_layers', 1, 4)
    gnn_units = []
    #dropouts = []
    max_units = 256
    for l in range(gnn_layers):
        postfix = '_l' + str(l)
        suggested_units = trial.suggest_int('gnn_units'+postfix, 10, max_units)
        gnn_units.append(suggested_units)
        max_units = suggested_units
        #dropouts.append(trial.suggest_float('dropouts'+postfix, 0.1, 0.3))

    mlp_layers =  trial.suggest_int('mlp_layers', 1, 4)
    # the number of units of the last gnn layer is the input dimension for the mlp
    mlp_units = [gnn_units[-1]]
    #dropouts = []
    for l in range(mlp_layers):
        postfix = '_l' + str(l)
        suggested_units = trial.suggest_int('mlp_units'+postfix, 5, max_units)
        mlp_units.append(suggested_units)
        max_units = suggested_units
        #dropouts.append(trial.suggest_float('dropouts'+postfix, 0.1, 0.3))

    # add output dimension 
    mlp_units.append(1)

    model_params =  {
        'conv_dim_list': gnn_units,
        'mlp_dim_list': mlp_units,
        #'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD']), 
        #'optimizer_lr': trial.suggest_float('optimizer_lr', 1e-5, 1e-1, log=True),
        'learning_rate': 0.001,
        # additional non-hyperparameter values
        'node_type_number': len(node2index),
        'padding_index': 0,
        'target_mean': transformer.target_mean, 
        'target_std': transformer.target_std,
    }

    logging.info('model params=' + str(model_params))

    model_instance = oscml.models.model_gnn.GNNSimple(**model_params)
    
    # fit on training set and calculate metric on validation set
    trainer_params = {
    }
    metric_value =  oscml.hpo.optunawrapper.fit(model_instance, train_dl, val_dl, trainer_params, trial)
    return metric_value


def fixed_trial():

    return {
        'gnn_layers': 2,
        'gnn_units_l0': 30,
        'gnn_units_l1': 20, 
        'mlp_layers': 3,
        'mlp_units_l0': 20,
        'mlp_units_l1': 10,
        'mlp_units_l2': 5,
        #'mlp_dropouts_l0': 0.2,
        #'mlp_dropouts_l1': 0.15,
        'optimizer': 'Adam',
        'optimizer_lr': 0.001,
        #'batch_size': 20
    }
   

if __name__ == '__main__':
    oscml.hpo.optunawrapper.start_hpo(init=init, objective=objective, metric='val_loss', direction='minimize')
    #oscml.hpo.optunawrapper.start_hpo(init=init, objective=objective, metric='val_loss', direction='minimize', fixed_trial=fixed_trial())