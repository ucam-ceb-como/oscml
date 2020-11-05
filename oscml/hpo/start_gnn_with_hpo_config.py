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


class Objective(object):
    # initialize the training data for objective function
    def __init__(self, config):
        # read hyperparameters from config file
        self.embedding_dimension = config['model_specific']['embedding_dimension']
        # graph conv
        self.graph_conv_number_layers = config['model_specific']['graph_conv']['number_of_layers']
        self.graph_conv_number_neurons = config['model_specific']['graph_conv']['number_of_neurons']
        self.graph_conv_dropout = config['model_specific']['graph_conv']['dropout']
        # mlp
        self.mlp_hidden_layers = config['model_specific']['mlp']['number_of_hidden_layers']
        self.mlp_number_neurons = config['model_specific']['mlp']['number_of_neurons']
        self.mlp_dropout = config['model_specific']['mlp']['dropout']
        # training
        self.learning_rate = config['training_specific']['learning_rate']
        self.optimiser = config['training_specific']['optimiser']

    def __call__(self, trial):

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
        gnn_layers = trial.suggest_int('gnn_layers', self.graph_conv_number_layers['lower'], self.graph_conv_number_layers['upper'])
        gnn_units = []
        #max_units = 256
        for l in range(gnn_layers):
            suggested_units = trial.suggest_int('gnn_units_{}'.format(l), self.graph_conv_number_neurons['lower'], self.graph_conv_number_neurons['upper'])
            #suggested_units = trial.suggest_int('gnn_units_{}'.format(l), 10, max_units)
            gnn_units.append(suggested_units)
            #max_units = suggested_units

        mlp_layers =  trial.suggest_int('mlp_hidden_layers', self.mlp_hidden_layers['lower'], self.mlp_hidden_layers['upper'])
        # the number of units of the last gnn layer is the input dimension for the mlp
        mlp_units = [gnn_units[-1]]
        mlp_dropout_rate = trial.suggest_float('mlp_dropout', self.mlp_dropout['lower'], self.mlp_dropout['upper'])
        mlp_dropouts = []
        for l in range(mlp_layers):
            suggested_units = trial.suggest_int('mlp_units_{}'.format(l), self.mlp_hidden_layers['lower'], self.mlp_hidden_layers['upper'])
            #suggested_units = trial.suggest_int('mlp_units_{}'.format(l), 5, max_units)
            mlp_units.append(suggested_units)
            #max_units = suggested_units
            mlp_dropouts.append(mlp_dropout_rate)

        # add output dimension 
        mlp_units.append(1)

        model_params =  {
            'conv_dim_list': gnn_units,
            'mlp_dim_list': mlp_units,
            'mlp_dropout_list': mlp_dropouts,
            'optimizer': trial.suggest_categorical('optimiser', self.optimiser), 
            'optimizer_lr': trial.suggest_loguniform('learning_rate', self.learning_rate['lower'], self.learning_rate['upper']),
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


if __name__ == '__main__':
    #metric='val_loss'
    metric='mse'
    oscml.hpo.optunawrapper.start_hpo(init=init, objective=Objective, metric='mse', direction='minimize')