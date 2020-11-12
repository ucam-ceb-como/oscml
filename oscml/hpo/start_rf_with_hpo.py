import logging

import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
import sklearn
import sklearn.datasets

import oscml.data.dataset
from oscml.utils.util import smiles2mol, concat


def init(user_attrs):
    # read data and preprocess, e.g. standarization, splitting into train, validation and test set
    src= user_attrs['src']
    dataset = user_attrs['dataset']
    return oscml.data.dataset.get_dataframes(dataset=dataset, src=src, train_size=283, test_size=30)


def objective(trial):

    user_attrs, init_attrs = oscml.hpo.optunawrapper.get_attrs(trial)

    metric = user_attrs['metric']
    #dataset = user_attrs['dataset']
    df_train = init_attrs[0]
    df_val = init_attrs[1]
    df_test = init_attrs[2]
    #transformer = init_attrs[3]


    df_train = pd.concat([df_train, df_val])

    #print(len(df_train), len(df_test))

    cross_validation = 10
    logging.info('cross_validation=' + str(cross_validation))

    fp_type = trial.suggest_categorical('type', ['morgan'])
    if fp_type == 'morgan':
        fp_params = {
            'nBits': trial.suggest_categorical('nBits', [128, 256, 512, 1024, 2048, 4096]), 
            'radius': trial.suggest_categorical('radius', [2, 3, 4, 5]), 
            'useChirality': trial.suggest_categorical('useChirality', [True, False]),
            'useBondTypes': trial.suggest_categorical('useBondTypes', [True, False]),
        }

    logging.info(concat('generating fingerprints, fp_type=', fp_type, ', fp_params=', fp_params))

    x_train, y_train = get_Morgan_fingerprints(df_train, fp_params, 'smiles', 'pce')
    #logging.info(concat('x_train=', len(x_train), 'y_train=', len(y_train)))
    x_test, y_test = get_Morgan_fingerprints(df_test, fp_params, 'smiles', 'pce')
    #logging.info(concat('x_test=', len(x_test), 'y_test=', len(y_test)))

    # see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
    # n_estimators=100, *, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)

    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
    max_samples = 10
    if bootstrap:
        max_samples_min = round(len(x_train) / 10)
        max_samples_max = round(len(x_train) / 2)
        max_samples = trial.suggest_int('max_samples', max_samples_min , max_samples_max)

    model_params =  {
        'n_estimators': trial.suggest_int('n_estimators', 16, 256),
        'max_depth': trial.suggest_int('max_depth', 10, 100),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 5),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 0.05, 0.1, 0.5, 1.0]),
        'bootstrap': bootstrap,
        'max_samples': max_samples,
    }

    logging.info(concat('starting cross validation for RF regressor, params=', model_params))
    regressor = sklearn.ensemble.RandomForestRegressor(**model_params, criterion=metric, random_state=0, verbose=0, n_jobs=1)
    scores = sklearn.model_selection.cross_val_score(regressor, x_train, y_train, cv=cross_validation)
    value = scores.mean()
    logging.info(concat('mean=', value, ', std=', scores.std(), ', scores=', scores))

    return value


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