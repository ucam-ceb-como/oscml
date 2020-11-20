import logging

import pandas as pd
import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
import sklearn

from oscml.utils.util import smiles2mol
from oscml.utils.util_config import set_config_param

def create(trial, config, df_train, df_val, df_test, training_params):

    x_column = config['dataset']['x_column'][0]
    y_column = config['dataset']['y_column'][0]

    # set model parameters from the config file
    #--------------------------------------
    model_params = {}
    for key, value in config['model']['model_specific'].items():
        model_params.update({key: set_config_param(trial=trial,param_name=key,param=value, all_params=model_params)})

    # set fingerprint parameters from the config file
    fp_params = {}
    for key, value in config['model']['fingerprint_specific'].items():
        fp_params.update({key: set_config_param(trial=trial,param_name=key,param=value, all_params=fp_params)})


    logging.info('model params=%s', model_params)
    logging.info('fingerprting params=%s', fp_params)

    # at the moment the only supported fingerprint choice is morgan
    fp_type = fp_params.pop('type',None)
    if fp_type=='morgan':
        get_fp = get_Morgan_fingerprints
    else:
        logging.exception('', exc_info=True)
        raise ValueError("Unknown fingerprint type '"+ fp_type+"'. Only 'morgan' fingerprints supported.")

    if training_params['cross_validation']:
        df_train = pd.concat([df_train, df_val])
        x_train, y_train = get_fp(df_train, fp_params, x_column, y_column)
        x_val = None
        y_val = None
    else:
        x_train, y_train = get_fp(df_train, fp_params, x_column, y_column)
        x_val, y_val = get_fp(df_val, fp_params, x_column, y_column)

    model = sklearn.ensemble.RandomForestRegressor(**model_params, criterion=training_params['criterion'], n_jobs=1, verbose=0, random_state=0)

    return model, x_train, y_train, x_val, y_val


def get_Morgan_fingerprints(df, params_morgan, columns_smiles, column_y):
    logging.info('generating Morgan fingerprint samples according to params=%s', params_morgan)
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