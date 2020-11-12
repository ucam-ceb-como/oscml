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
    dataset = user_attrs['dataset']
    info = oscml.data.dataset.get_dataset_info(dataset)

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
            'type': 'morgan',
            'nBits': trial.suggest_categorical('nBits', [128, 256, 512, 1024, 2048, 4096]),
            'radius': trial.suggest_categorical('radius', [2, 3, 4, 5]),
            'useChirality': trial.suggest_categorical('useChirality', [True, False]),
            'useBondTypes': trial.suggest_categorical('useBondTypes', [True, False]),
        }

    x_train,  y_train, scaler_svr_physical_data = oscml.models.model_kernel.preprocess_data_phys_and_struct(df_train, fp_params, train_size=1, column_smiles=info.column_smiles,
                                columns_phys=None, column_y=info.column_target)

    logging.info(concat('generating fingerprints, fp_type=', fp_type, ', fp_params=', fp_params))

    model_params =  {
        'kernel': 'rbf_kernel_phys_and_struct',
        'C': trial.suggest_float('C',0.1,10.0),
        'epsilon': trial.suggest_float('epsilon',0.01,2.0),
        'gamma_structural': trial.suggest_float('gamma_structural',0.1,10.0)
    }

    logging.info(concat('starting cross validation for SVR regressor, params=', model_params))
    model = oscml.models.model_kernel.SVRWrapper(**model_params)

    all_scores = sklearn.model_selection.cross_validate(model, x_train, y_train, cv=cross_validation,
                scoring='neg_mean_squared_error',  n_jobs=1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs', return_train_score=True, return_estimator=True , error_score='raise')

    #print(all_scores)
    for phase in ['train', 'test']:
        scores = all_scores[phase + '_score']
        mean = scores.mean()
        std = scores.std()
        logging.info(concat(phase, ': mean', mean, ', std=', std, ', scores=', scores))

    objective_value = - mean # from test scores

    # check the objective value
    """
    score_sum = 0.
    for reg in all_scores['estimator']:
        y_pred = reg.predict(x_train)
        metrics = oscml.utils.util.calculate_metrics(y_train, y_pred)
        score_sum += metrics['mse']
    mean_score = score_sum / len(all_scores['estimator'])
    logging.info('mean score sum on entire train set=' + str(mean_score))
    """

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
            objective=objective,
            metric='mse',
            direction='minimize',
            fixed_trial_params=fixed_trial())

if __name__ == '__main__':
    start()
