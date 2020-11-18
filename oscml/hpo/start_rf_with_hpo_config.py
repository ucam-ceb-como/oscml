import logging

import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
import sklearn
import sklearn.datasets

import oscml.data.dataset
import oscml.hpo.optunawrapper
from oscml.utils.util import smiles2mol, concat
import oscml.utils.util_sklearn


def init(user_attrs):
    # read data and preprocess, e.g. standarization, splitting into train, validation and test set
    src= user_attrs['src']
    metric = user_attrs['metric']
    if not metric == 'mse':
        message = 'only metric mse is supported, but metric was set to ' + str(metric)
        raise RuntimeError(message)
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
        self.number_of_estimators = config['model_specific']['number_of_estimators']
        self.max_depth_of_each_tree = config['model_specific']['max_depth_of_each_tree']
        self.min_samples_split = config['model_specific']['min_samples_split']
        self.min_samples_leaf = config['model_specific']['min_samples_leaf']
        self.max_features = config['model_specific']['max_features']
        self.bootstraping = parseCategoricalBoolean(config['model_specific']['bootstraping'])
        self.max_nr_of_samples = config['model_specific']['max_nr_of_samples']

        # fingerprint
        self.type = config['fingerprints']['type']
        self.nr_of_bits = config['fingerprints']['nr_of_bits']
        self.radius = config['fingerprints']['radius']
        self.use_bond_type = parseCategoricalBoolean(config['fingerprints']['use_bond_type'])
        self.use_chirality = parseCategoricalBoolean(config['fingerprints']['use_chirality'])

        # training
        #self.number_of_cross_validation = config['training_specific']['number_of_cross_validations']


    def __call__(self, trial):

        user_attrs, init_attrs = oscml.hpo.optunawrapper.get_attrs(trial)

        metric = 'mse'  # user_attrs['metric']
        cv = user_attrs['cv']
        dataset = user_attrs['dataset']
        info = oscml.data.dataset.get_dataset_info(dataset)

        df_train = init_attrs[0]
        df_val = init_attrs[1]
        #df_test = init_attrs[2]
        #transformer = init_attrs[3]

        fp_type = trial.suggest_categorical('type', self.type)
        if fp_type == 'morgan':
            fp_params = {
                'nBits': trial.suggest_categorical('nBits', self.nr_of_bits),
                'radius': trial.suggest_categorical('radius', self.radius),
                'useChirality': trial.suggest_categorical('useChirality', self.use_bond_type),
                'useBondTypes': trial.suggest_categorical('useBondTypes', self.use_chirality),
            }

        logging.info(concat('generating fingerprints, fp_type=', fp_type, ', fp_params=', fp_params))

        if cv:
            df_train = pd.concat([df_train, df_val])
            x_train, y_train = get_Morgan_fingerprints(df_train, fp_params, info.column_smiles, info.column_target)
            x_val = None
            y_val = None
        else:
            x_train, y_train = get_Morgan_fingerprints(df_train, fp_params, info.column_smiles, info.column_target)
            x_val, y_val = get_Morgan_fingerprints(df_val, fp_params, info.column_smiles, info.column_target)

        # see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
        # n_estimators=100, *, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)

        bootstrap = trial.suggest_categorical('bootstrap', self.bootstraping)
        max_samples = 10
        if bootstrap:
            max_samples_min = self.max_nr_of_samples['upper']  # round(len(x_train) / 10)
            max_samples_max = round(len(x_train) / 2)
            max_samples = trial.suggest_int('max_samples', max_samples_min, max_samples_max)

        model_params = {
            'n_estimators': trial.suggest_int('n_estimators', self.number_of_estimators['lower'], self.number_of_estimators['upper']),
            'max_depth': trial.suggest_int('max_depth', self.max_depth_of_each_tree['lower'], self.max_depth_of_each_tree['upper']),
            'min_samples_split': trial.suggest_int('min_samples_split', self.min_samples_split['lower'], self.min_samples_split['upper']),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', self.min_samples_leaf['lower'], self.min_samples_leaf['upper']),
            'max_features': trial.suggest_categorical('max_features', self.max_features),
            'bootstrap': bootstrap,
            'max_samples': max_samples,
        }

        # The default scoring value is None. In this case, the estimator’s default scorer (if available) is used.
        # The score function of RandomForestRegressor returns the coefficient of determination R^2 of the prediction,
        # see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
        # Therefore, we use here ‘neg_mean_squared_error’ and multiply with -1 to obtain MSE as objective value.
        logging.info(concat('starting cross validation for RF regressor, params=', model_params))
        model = sklearn.ensemble.RandomForestRegressor(**model_params, criterion=metric, random_state=0, verbose=0,
                                                                   n_jobs=1)

        objective_value = oscml.utils.util_sklearn.train_and_test(x_train, y_train, x_val, y_val, model,
                                                                  cross_validation=cv, metric=metric)
        logging.info(concat('objective value', objective_value))

        return objective_value

def get_Morgan_fingerprints(df, params_morgan, columns_smiles, column_y):
    logging.info('generating Morgan fingerprint samples according to params=' + str(params_morgan))
    x = []
    y = []
    for i in range(len(df)):
        smiles = df.iloc[i][columns_smiles]
        m = smiles2mol(smiles)
        fingerprint = rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect(m, **params_morgan)
        x.append(fingerprint)
        pce = df.iloc[i][column_y]
        y.append(pce)

    return x, y

def fixed_trial():
    return {
        'n_estimators': 50,
        'max_depth': 20,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 0.1,
        'bootstrap': True,
        'max_samples': 50,
        # fingerprint params
        'type': 'morgan',
        'nBits': 128,
        'radius': 2,
        'useChirality': False,
        'useBondTypes': True,
    }

def start():
    return oscml.hpo.optunawrapper.start_hpo(
        init=init,
        objective=Objective,
        metric='mse',
        direction='minimize',
        fixed_trial_params=fixed_trial())

if __name__ == '__main__':
    start()