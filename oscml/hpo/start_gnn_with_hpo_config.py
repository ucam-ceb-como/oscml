import logging
import types

import oscml.data.dataset
import oscml.data.dataset_cep
import oscml.data.dataset_hopv15
import oscml.hpo.optunawrapper
import oscml.models.model_gnn


def init(user_attrs):

    # read data and preprocess, e.g. standarization, splitting into train, validation and test set
    src = user_attrs['src']
    dataset = user_attrs['dataset']
    return oscml.data.dataset.get_dataframes(dataset=dataset, src=src, train_size=283, test_size=30)

def parseCategoricalBoolean(config_list):
    boolean_list = []
    for s in config_list:
        if s == "True":
            boolean_list.append(True)
        else:
            boolean_list.append(False)

    return boolean_list

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
        self.weight_decay = config['training_specific']['weight_decay']
        self.momentum = config['training_specific']['momentum']
        self.nesterov = parseCategoricalBoolean(config['training_specific']['nesterov'])

    def __call__(self, trial):

        user_attrs, init_attrs = oscml.hpo.optunawrapper.get_attrs(trial)

        dataset = user_attrs['dataset']
        df_train = init_attrs[0]
        df_val = init_attrs[1]
        df_test = None
        transformer = init_attrs[3]

        info = oscml.data.dataset.get_dataset_info(dataset)
        node_type_number = len(info.node_types)

        # define data loader and params
        train_dl, val_dl, test_dl = oscml.models.model_gnn.get_dataloaders(dataset, df_train, df_val, df_test,
                                                                         transformer, batch_size=20)


        # define model and params
        embedding_dim = trial.suggest_int('embedding_dim', self.embedding_dimension['lower'], self.embedding_dimension['upper'])
        conv_dims = []
        conv_layers = trial.suggest_int('conv_layers', self.graph_conv_number_layers['lower'], self.graph_conv_number_layers['upper'])
        max_units = 256
        for l in range(conv_layers):
            suggested_units = trial.suggest_int('conv_dims_{}'.format(l), self.graph_conv_number_neurons['lower'], max_units)
            conv_dims.append(suggested_units)
            max_units = suggested_units

        mlp_layers =  trial.suggest_int('mlp_hidden_layers', self.mlp_hidden_layers['lower'], self.mlp_hidden_layers['upper'])
        # the number of units of the last gnn layer is the input dimension for the mlp
        mlp_units = []
        mlp_dropout_rate = trial.suggest_float('mlp_dropout', self.mlp_dropout['lower'], self.mlp_dropout['upper'])
        mlp_dropouts = []
        for l in range(mlp_layers):
            suggested_units = trial.suggest_int('mlp_units_{}'.format(l), self.mlp_hidden_layers['lower'], max_units)
            #suggested_units = trial.suggest_int('mlp_units_{}'.format(l), 5, max_units)
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
            'node_type_number': node_type_number,
            'padding_index': 0,
            'target_mean': transformer.target_mean, 
            'target_std': transformer.target_std,
        }

        logging.info('model params=' + str(model_params))

        # torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        # torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
        # torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        name = trial.suggest_categorical('name', self.optimiser)
        optimiser = {
            'name': name,
            'lr': trial.suggest_loguniform('lr', self.learning_rate['lower'], self.learning_rate['upper']),
            'weight_decay': trial.suggest_uniform('weight_decay', self.weight_decay['lower'], self.weight_decay['upper'])
        }
        if name in ['RMSprop', 'SGD']:
            optimiser['momentum'] = trial.suggest_uniform('momentum', self.momentum['lower'], self.momentum['upper'])
        if name == 'SGD':
            optimiser['nesterov'] = trial.suggest_categorical('nesterov', self.nesterov)

        model = oscml.models.model_gnn.SimpleGNN(**model_params, optimizer=optimiser)
        
        # fit on training set and calculate metric on validation set
        trainer_params = {}
        metric_value = oscml.hpo.optunawrapper.fit(model, train_dl, val_dl, trainer_params, trial)
        return metric_value

def resume(ckpt, src, log_dir, dataset, epochs, metric):

    model_class = oscml.models.model_gnn.GNNSimple
    model = model_class.load_from_checkpoint(ckpt)

    info = oscml.data.dataset.get_dataset_info(dataset)

    transformer = oscml.data.dataset.DataTransformer(info.column_target, model.target_mean,
                                                     model.target_std, info.column_smiles)

    df_train, df_val, df_test, _ = oscml.data.dataset.get_dataframes(dataset, src=src, train_size=283, test_size=30)
    train_dl, val_dl, test_dl = oscml.models.model_gnn.get_dataloaders(dataset, df_train, df_val, df_test, transformer,
                                                                       batch_size=20)

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
        'embedding_dim':30,
        'conv_layers': 2,
        'conv_dims_0': 30,
        'conv_dims_1': 20,
        'mlp_layers': 3,
        'mlp_units_0': 20,
        'mlp_units_1': 10,
        'mlp_units_2': 5,
        'mlp_dropout': 0.2,
        'name': 'Adam',             # Adam, SGD, RMSProp
        'lr': 0.001,
        'momentum': 0,              # SGD and RMSProp only
        'weight_decay': 0,
        'nesterov': False,          # SGD only
        #'batch_size': 20
    }

def start():
    return oscml.hpo.optunawrapper.start_hpo(
        init=init,
        objective=Objective,
        metric='val_loss',
        direction='minimize',
        fixed_trial_params=fixed_trial(),
        resume=resume)

if __name__ == '__main__':
    start()