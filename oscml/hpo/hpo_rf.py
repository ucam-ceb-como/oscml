import logging
import sklearn
import oscml.data.dataset
import pandas as pd
from oscml.utils.util_config import set_config_param
from oscml.utils.util import smiles2mol, concat
import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
from oscml.utils.util_config import set_config_param

def create(trial, config, df_train, df_val, df_test, training_params, dataset):

    info = oscml.data.dataset.get_dataset_info(dataset)
    node_type_number = len(info.node_types)

    # define model and params
    # copy model params to another dictionary, as we may need to modify some of its values
    model_specific = config['model']['model_specific'].copy()
    fp_specific = config['model']['fingerprint_specific'].copy()
    training_specific = config['training'].copy()

    # model_params
    n_estimators = set_config_param(trial=trial,param_name='n_estimators',param=model_specific['n_estimators'])
    max_depth = set_config_param(trial=trial,param_name='max_depth',param=model_specific['max_depth'])
    min_samples_split = set_config_param(trial=trial,param_name='min_samples_split',param=model_specific['min_samples_split'])
    min_samples_leaf = set_config_param(trial=trial,param_name='min_samples_leaf',param=model_specific['min_samples_leaf'])
    max_features = set_config_param(trial=trial,param_name='max_features',param=model_specific['max_features'])
    bootstrap = set_config_param(trial=trial,param_name='bootstrap',param=model_specific['bootstrap'])
    max_samples = set_config_param(trial=trial,param_name='max_samples',param=model_specific['max_samples'])
    cross_validation = set_config_param(trial=trial,param_name='cross_validation',param=model_specific['cross_validation'])
    # fingerprint params
    fp_type = set_config_param(trial=trial,param_name='fp_type',param=fp_specific['type'])
    fp_nBits = set_config_param(trial=trial,param_name='fp_nBits',param=fp_specific['nBits'])
    fp_radius = set_config_param(trial=trial,param_name='fp_radius',param=fp_specific['radius'])
    fp_use_chirality = set_config_param(trial=trial,param_name='fp_use_chirality',param=fp_specific['useChirality'])
    fp_use_bond_types = set_config_param(trial=trial,param_name='fp_use_bond_types',param=fp_specific['useBondTypes'])

    model_params =  {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
        'bootstrap': bootstrap,
        'max_samples': max_samples
    }
    fp_params = {
        # fingerprint params
        'nBits': fp_nBits,
        'radius': fp_radius,
        'useChirality': fp_use_chirality,
        'useBondTypes': fp_use_bond_types
    }

    if cross_validation:
        df_train = pd.concat([df_train, df_val])
        x_train, y_train = get_Morgan_fingerprints(df_train, fp_params, info.column_smiles, info.column_target)
        x_val = None
        y_val = None
    else:
        x_train, y_train = get_Morgan_fingerprints(df_train, fp_params, info.column_smiles, info.column_target)
        x_val, y_val = get_Morgan_fingerprints(df_val, fp_params, info.column_smiles, info.column_target)


    logging.info('model params=%s', model_params)
    model = sklearn.ensemble.RandomForestRegressor(**model_params, **training_params, n_jobs=1, verbose=0, random_state=0)

    return model, x_train, y_train, x_val, y_val, cross_validation


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