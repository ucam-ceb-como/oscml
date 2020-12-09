import logging

import sklearn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import ShuffleSplit

import oscml.utils.util


def calculate_mean_prediction(regressors, x, y):

    # ensemble learning
    y_sum = None
    # average mse
    mse_sum = None

    for reg in regressors:
        y_pred = reg.predict(x)
        metrics = oscml.utils.util.calculate_metrics(y, y_pred)
        if y_sum is None:
            y_sum = y_pred
            mse_sum = metrics['mse']
        else:
            y_sum += y_pred
            mse_sum += metrics['mse']
    y_mean = y_sum / len(regressors)
    mse_mean = mse_sum / len(regressors)
    metrics_y_mean = oscml.utils.util.calculate_metrics(y, y_mean)
    return mse_mean, metrics_y_mean

def get_length(z):
    if z is None:
        return None
    return len(z)

def train_and_test(x_train, y_train, x_val, y_val, x_test, y_test, model, cross_validation, metric, log_dir,
                   seed=None, trial_number=None, best_trial_retrain=False, transformer=None, inverse=False, regression_plot=False):

    if best_trial_retrain:
        log_head = '[Best trial retrain - Trial ' + str(trial_number) + ']'
    else:
        if not cross_validation:
            log_head = '[Trial ' + str(trial_number) + ']'
        else:
            log_head = '[Trial '+ str(trial_number) + ' - ' + str(cross_validation) + ' fold cross-validation]'

    logging.info('%s fitting for %s', log_head, model)
    logging.info('%s sample size = %s %s %s %s %s %s', log_head, len(x_train), len(y_train), get_length(x_val), get_length(y_val), get_length(x_test), get_length(y_test))
    logging.info('%s cross_validation = %s', log_head, cross_validation)

    if best_trial_retrain:
        dirpath = log_dir + '/trial_' + str(trial_number) + '/'
        Path(dirpath).mkdir(parents=True, exist_ok=True)

    if cross_validation:
        if not best_trial_retrain:
            # use cross_validate instead of cross_val_score to get more information about scores
            # only use 1 CPU (n_jobs=1)
            all_scores = sklearn.model_selection.cross_validate(model, x_train, y_train, cv=cross_validation,
                        scoring='neg_mean_squared_error',  n_jobs=1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs', return_train_score=True, return_estimator=True , error_score='raise')
            for phase in ['train', 'test']:
                scores = all_scores[phase + '_score']
                mean = scores.mean()
                std = scores.std()
                logging.info('%s %s: mean %s, std=%s, scores=%s', log_head, phase, mean, std, scores)

            objective_value = - mean # from test scores

            #print('MY', all_scores)
            assert cross_validation == len(all_scores['estimator'])

            regressors = all_scores['estimator']
            mse_mean_train, result_train = calculate_mean_prediction(regressors, x_train, y_train)
            logging.info('%s mean mse train=%s, ensemble train result=%s', log_head, mse_mean_train, result_train)
            mse_mean_test, result_test = calculate_mean_prediction(regressors, x_test, y_test)
            logging.info('%s mean mse test=%s, ensemble test result=%s', log_head, mse_mean_test, result_test)

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
        else:
            rs = ShuffleSplit(n_splits=1, test_size=1 / cross_validation, random_state=seed + 1)
            rs.get_n_splits(x_train)
            for train_index, val_index in rs.split(x_train):
                result = model_retraining(model, metric, x_train[train_index], y_train[train_index],
                                          x_train[val_index], y_train[val_index],
                                          x_test, y_test,
                                          dirpath, transformer, inverse, regression_plot, log_head)
                objective_value = result[metric]
    else:
        if not best_trial_retrain:
            model.fit(x_train, y_train)
    
            """
            y_pred_train = model.predict(x_train)
            y_pred_val = model.predict(x_val)
            y_pred_test = model.predict(x_test)
        
            if metric == 'mse':
                result_train = oscml.utils.util.calculate_metrics(y_train, y_pred_train)
                result_val = oscml.utils.util.calculate_metrics(y_val, y_pred_val)
                result_test = oscml.utils.util.calculate_metrics(y_test, y_pred_test)
            else: # accuracy
                result_train = {'accuracy': sklearn.metrics.accuracy_score(y_train, y_pred_train)}
                result_val = {'accuracy': sklearn.metrics.accuracy_score(y_val, y_pred_val)}
                result_test = {'accuracy': sklearn.metrics.accuracy_score(y_test, y_pred_test)}
                
            logging.info('train result=%s', result_train)
            logging.info('val result=%s', result_val)
            logging.info('test result=%s', result_test)
            """

            calculate_metrics(model, x_train, y_train, metric, 'train', log_head)
            objective_value = 0.
            if x_val is not None:
                result = calculate_metrics(model, x_val, y_val, metric, 'val', log_head)
                objective_value = result[metric]
            if x_test is not None:
                result = calculate_metrics(model, x_test, y_test, metric, 'test', log_head)

        else:
            result = model_retraining(model, metric, x_train, y_train, x_val, y_val, x_test, y_test,
                                      dirpath, transformer, inverse, regression_plot, log_head)
            objective_value = result[metric]


    return objective_value

def calculate_metrics(model, x, y, metric, ml_phase, log_head):
    y_pred = model.predict(x)
    if metric == 'mse':
        result = oscml.utils.util.calculate_metrics(y, y_pred)
    elif metric == 'all':
        result = {'phase': ml_phase}
        result.update(oscml.utils.util.calculate_metrics(y, y_pred))
        return result, y_pred
    else: # accuracy
        result = {'accuracy': sklearn.metrics.accuracy_score(y, y_pred)}
        
    logging.info('%s %s result=%s', log_head, ml_phase, result)
    return result


def log_and_plot(model, x_train, y_train, x_val, y_val, x_test, y_test, dirpath, transformer=None,
                 inverse=False, regression_plot=False, log_head=None):
    index_ml = ['training set', 'validation set', 'test set']
    x_ml = [x_train, x_val, x_test]
    y_ml = [y_train, y_val, y_test]
    results_metric = []
    for index_, x_, y_ in zip(index_ml, x_ml, y_ml):
        test_results, y_pred = calculate_metrics(model, x_, y_, 'all', index_, log_head)
        results_metric.append(test_results)

        if not inverse:
            pred_df = pd.DataFrame(list(standard_score_transform(transformer, np.array(y_))),
                                   columns=['Measured PCE'])
            pred_df['Predicted PCE'] = list(standard_score_transform(transformer, np.array(y_pred)))
        else:
            pred_df = pd.DataFrame(y_, columns=['Measured PCE'])
            pred_df['Predicted PCE'] = y_pred

        pred_df.to_csv(dirpath + 'predictions_{}.csv'.format(index_.replace(' ', '_')))
    pd.DataFrame(results_metric).to_csv(dirpath + 'best_trial_retrain_model_result.csv')

    if regression_plot:
        oscml.visualization.util_sns_plot.prediction_plot(dirpath, dirpath + 'predictions_training_set.csv',
                                                          dirpath + 'predictions_validation_set.csv',
                                                          dirpath + 'predictions_test_set.csv')


def model_retraining(model, metric, x_train, y_train, x_val, y_val, x_test, y_test, dirpath, transformer=None, inverse=False,
                     regression_plot=False, log_head=None):
    model.fit(x_train, y_train)
    log_and_plot(model, x_train, y_train, x_val, y_val, x_test, y_test, dirpath, transformer=transformer,
                 inverse=inverse, regression_plot=regression_plot, log_head=log_head)
    return calculate_metrics(model, x_val, y_val, metric, 'val', log_head)


def standard_score_transform(transformer, y):
    y_transform = (y - transformer.target_mean) / transformer.target_std
    return y_transform