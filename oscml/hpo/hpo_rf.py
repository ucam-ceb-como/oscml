import logging
import sklearn
import oscml.data.dataset




from oscml.utils.util_config import set_config_param, set_config_param_list

def create(trial, config, df_train, df_val, df_test, optimizer, transformer, dataset):

    info = oscml.data.dataset.get_dataset_info(dataset)
    node_type_number = len(info.node_types)

    #dataloaders
    #train_dl, val_dl, test_dl = oscml.models.model_gnn.get_dataloaders(dataset, df_train, df_val, df_test,
    #        transformer, batch_size=250)

    # define model and params
    # copy model params to another dictionary, as we may need to modify some of its values
    model_specific = config['model']['model_specific'].copy()
    fp_specific = config['fingerprint'].copy()


    n_estimators = set_config_param(trial=trial,param_name='n_estimators',param=model_specific['n_estimators'])
    max_depth = set_config_param(trial=trial,param_name='max_depth',param=model_specific['max_depth'])
    min_samples_split = set_config_param(trial=trial,param_name='min_samples_split',param=model_specific['min_samples_split'])
    min_samples_leaf = set_config_param(trial=trial,param_name='min_samples_leaf',param=model_specific['min_samples_leaf'])
    max_features = set_config_param(trial=trial,param_name='max_features',param=model_specific['max_features'])
    bootstrap = set_config_param(trial=trial,param_name='bootstrap',param=model_specific['bootstrap'])
    max_samples = set_config_param(trial=trial,param_name='max_samples',param=model_specific['max_samples'])
    cross_validation = set_config_param(trial=trial,param_name='cross_validation',param=model_specific['cross_validation'])
    #
    fp_type = set_config_param(trial=trial,param_name='fp_type',param=fp_specific['fp_type'])
    fp_nBits = set_config_param(trial=trial,param_name='fp_nBits',param=fp_specific['fp_nBits'])
    fp_radius = set_config_param(trial=trial,param_name='fp_radius',param=fp_specific['fp_radius'])
    fp_use_chirality = set_config_param(trial=trial,param_name='fp_use_chirality',param=fp_specific['fp_use_chirality'])
    fp_use_bond_types = set_config_param(trial=trial,param_name='fp_use_bond_types',param=fp_specific['fp_use_bond_types'])

    model_params =  {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
        'bootstrap': bootstrap,
        'max_samples': max_samples,
    }
    #{    #'cross_validation': cross_validation,
    #    # fingerprint params
    #    'type': fp_type,
    #    'nBits': fp_nBits,
    #    'radius': fp_radius,
    #    'useChirality': fp_use_chirality,
    #    'useBondTypes': fp_use_bond_types,
    #}

    logging.info('model params=%s', model_params)

    model = sklearn.ensemble.RandomForestRegressor(**model_params, criterion='mse', random_state=0, verbose=0, n_jobs=1)
    #optimizer=optimizer) # criterion=metric, random_state=0, verbose=0,n_jobs=1)

    return model #, train_dl, val_dl, test_dl


class Objective(object):
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