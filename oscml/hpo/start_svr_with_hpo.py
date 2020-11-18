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
import oscml.utils.util_sklearn
from oscml.utils.util import smiles2mol, concat
import oscml.models.model_kernel

def init(user_attrs):
    # read data and preprocess, e.g. standarization, splitting into train, validation and test set
    src= user_attrs['src']
    metric = user_attrs['metric']
    if not metric == 'mse':
        message = 'only metric mse is supported, but metric was set to ' + str(metric)
        raise RuntimeError(message)
    dataset = user_attrs['dataset']
    return oscml.data.dataset.get_dataframes(dataset=dataset, src=src, train_size=283, test_size=30)

def objective(trial):

    user_attrs, init_attrs = oscml.hpo.optunawrapper.get_attrs(trial)

    metric = 'mse' # user_attrs['metric']
    cv = user_attrs['cv']
    dataset = user_attrs['dataset']
    info = oscml.data.dataset.get_dataset_info(dataset)

    df_train = init_attrs[0]
    df_val = init_attrs[1]
    #df_test = init_attrs[2]
    #transformer = init_attrs[3] 

    fp_type = trial.suggest_categorical('type', ['morgan'])
    if fp_type == 'morgan':
        fp_params = {
            'type': 'morgan',
            'nBits': trial.suggest_categorical('nBits', [128, 256, 512, 1024, 2048, 4096]),
            'radius': trial.suggest_categorical('radius', [1, 2, 3, 4, 5, 6]),
            'useChirality': trial.suggest_categorical('useChirality', [True, False]),
            'useBondTypes': trial.suggest_categorical('useBondTypes', [True, False]),
        }

    logging.info(concat('generating fingerprints, fp_type=', fp_type, ', fp_params=', fp_params))

    if cv:
        df_train = pd.concat([df_train, df_val])
        x_train,  y_train, scaler_svr_physical_data = oscml.models.model_kernel.preprocess_data_phys_and_struct(df_train, fp_params, train_size=1, column_smiles=info.column_smiles,
                                columns_phys=None, column_y=info.column_target)
        x_val = None
        y_val = None
    else:
        x_train,  y_train, scaler_svr_physical_data = oscml.models.model_kernel.preprocess_data_phys_and_struct(df_train, fp_params, train_size=1, column_smiles=info.column_smiles,
                                columns_phys=None, column_y=info.column_target)

        x_val,  y_val, scaler_svr_physical_data = oscml.models.model_kernel.preprocess_data_phys_and_struct(df_val, fp_params, train_size=1, column_smiles=info.column_smiles,
                                columns_phys=None, column_y=info.column_target, scaler_svr_physical_data=scaler_svr_physical_data)

    model_params =  {
        'kernel': 'rbf_kernel_phys_and_struct',
        'C': trial.suggest_loguniform('C',0.1,20.0),
        'epsilon': trial.suggest_loguniform('epsilon',0.0001,1.0),
        'gamma_structural': trial.suggest_loguniform('gamma_structural',0.001,20.0)
    }

    logging.info(concat('starting SVR regressor, params=', model_params))
    model = oscml.models.model_kernel.SVRWrapper(**model_params)

    objective_value = oscml.utils.util_sklearn.train_and_test(x_train, y_train, x_val, y_val, model, cross_validation=cv, metric=metric)
    logging.info(concat('objective value', objective_value))

    return objective_value


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
            objective=objective,
            metric='mse',
            direction='minimize',
            fixed_trial_params=fixed_trial())

if __name__ == '__main__':
    start()
