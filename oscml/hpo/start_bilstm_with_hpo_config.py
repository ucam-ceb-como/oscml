import logging
import types

import oscml.data.dataset
import oscml.data.dataset_cep
import oscml.data.dataset_hopv15
import oscml.hpo.optunawrapper
import oscml.models.model_bilstm


def init(user_attrs):
    # read data and preprocess, e.g. standarization, splitting into train, validation and test set
    src = user_attrs['src']
    dataset = user_attrs['dataset']
    return oscml.data.dataset.get_dataframes(dataset=dataset, src=src, train_size=283, test_size=30)


class Objective(object):
    # initialize the training data for objective function
    def __init__(self, config):
        # read hyperparameters from config file
        self.embedding_dimension = config['model_specific']['embedding_dimension']
        self.dropout = config['model_specific']['dropout']
        # mlp
        self.mlp_hidden_layers = config['model_specific']['mlp']['number_of_hidden_layers']
        self.mlp_number_neurons = config['model_specific']['mlp']['number_of_neurons']
        self.mlp_dropout = config['model_specific']['mlp']['dropout']
        # training
        self.learning_rate = config['training_specific']['learning_rate']
        self.optimiser = config['training_specific']['optimiser']
        self.weight_decay = config['training_specific']['weight_decay']
        self.batch_size = config['training_specific']['batch_size']

    def __call__(self, trial):

        user_attrs, init_attrs = oscml.hpo.optunawrapper.get_attrs(trial)

        dataset = user_attrs['dataset']
        df_train = init_attrs[0]
        df_val = init_attrs[1]
        df_test = None
        transformer = init_attrs[3]

        info = oscml.data.dataset.get_dataset_info(dataset)
        max_sequence_length = info.max_sequence_length
        number_subgraphs = info.number_subgraphs()
        train_dl, val_dl, _ = oscml.models.model_bilstm.get_dataloaders(dataset, df_train, df_val, df_test,
            transformer, batch_size=20, max_sequence_length=max_sequence_length)

        # define models and params
        subgraph_embedding_dim = trial.suggest_int('subgraph_embedding_dim', self.embedding_dimension['lower'],
                                                   self.embedding_dimension['upper'])
        lstm_hidden_dim = subgraph_embedding_dim
        mlp_layers = 1 + trial.suggest_int('mlp_hidden_layers', self.mlp_hidden_layers['lower'], self.mlp_hidden_layers['upper'])
        mlp_units = []
        mlp_dropout_rate = trial.suggest_float('mlp_dropout', self.mlp_dropout['lower'], self.mlp_dropout['upper'])
        mlp_dropouts = []
        max_units = 2 * lstm_hidden_dim
        for l in range(mlp_layers):
            suggested_units = trial.suggest_int('mlp_units_{}'.format(l), self.mlp_number_neurons['lower'], max_units)
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
            # additional non-hyperparameter values
            'number_of_subgraphs': number_subgraphs,
            'padding_index': 0,
            'target_mean': transformer.target_mean,
            'target_std': transformer.target_std,
        }

        logging.info('model params=' + str(model_params))

        name = trial.suggest_categorical('name', self.optimiser)
        optimizer = {'name': name,
                     'lr': trial.suggest_loguniform('lr', self.learning_rate['lower'], self.learning_rate['upper']),
                     'weight_decay': trial.suggest_uniform('weight_decay', self.weight_decay['lower'], self.weight_decay['upper'])
                     }
        if name in ['RMSprop', 'SGD']:
            optimizer['momentum'] = trial.suggest_uniform('momentum', 0, 0.01)
        if name == 'SGD':
            optimizer['nesterov'] = trial.suggest_categorical('nesterov', [True, False])

        model = oscml.models.model_bilstm.BiLstmForPce(**model_params, optimizer=optimizer)

        # fit on training set and calculate metric on validation set
        trainer_params = {}
        metric_value =  oscml.hpo.optunawrapper.fit(model, train_dl, val_dl, trainer_params, trial)
        return metric_value


def resume(ckpt, src, log_dir, dataset, epochs, metric):
    model_class = oscml.models.model_bilstm.BiLstmForPce
    model = model_class.load_from_checkpoint(ckpt)

    info = oscml.data.dataset.get_dataset_info(dataset)

    transformer = oscml.data.dataset.DataTransformer(info.column_target, model.target_mean,
                                                     model.target_std, info.column_smiles)

    max_sequence_length = info.max_sequence_length
    df_train, df_val, df_test, _ = oscml.data.dataset.get_dataframes(dataset, src=src, train_size=283, test_size=30)
    train_dl, val_dl, test_dl = oscml.models.model_bilstm.get_dataloaders(dataset, df_train, df_val, df_test,
                                                                          transformer, batch_size=20,
                                                                          max_sequence_length=max_sequence_length)

    if epochs > 0:
        trainer_params = {}
        result = oscml.hpo.optunawrapper.fit_or_test(model, train_dl, val_dl, None, trainer_params,
                                                     epochs, metric, log_dir)

    else:
        trainer_params = {}
        result = oscml.hpo.optunawrapper.fit_or_test(model, None, None, test_dl, trainer_params,
                                                     epochs, metric, log_dir)

    return result

def fixed_trial():
    return {
        'subgraph_embedding_dim': 128,
        #'lstm_hidden_dim': 128,  # currently set to subgraph_embedding_dim
        'mlp_layers': 3,
        'mlp_units_0': 64,
        'mlp_units_1': 32,
        'mlp_units_2': 32,       
        'mlp_dropout': 0.1,
        'name': 'Adam',             # Adam, SGD, RMSProp
        'lr': 0.001,
        'momentum': 0,              # SGD and RMSProp only
        'weight_decay': 0,
        'nesterov': False,          # SGD only
        #'batch_size': 250
    }

def start():
    return oscml.hpo.optunawrapper.start_hpo(
        init=init,
        objective=Objective,
        metric='val_loss',
        direction='minimize',
        fixed_trial_params=fixed_trial(),
        resume=resume
    )

if __name__ == '__main__':
    start()