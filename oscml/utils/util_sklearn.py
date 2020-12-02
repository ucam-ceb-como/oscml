import logging

import sklearn

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

def train_and_test(x_train, y_train, x_val, y_val, x_test, y_test, model, cross_validation, metric):
       
    logging.info('fitting for %s', model)
    logging.info('sample size = %s %s %s %s %s %s', len(x_train), len(y_train), get_length(x_val), get_length(y_val), get_length(x_test), get_length(y_test))
    logging.info('cross_validation = %s', cross_validation)
    
    if cross_validation:
        # use cross_validate instead of cross_val_score to get more information about scores
        # only use 1 CPU (n_jobs=1)
        all_scores = sklearn.model_selection.cross_validate(model, x_train, y_train, cv=cross_validation,
                    scoring='neg_mean_squared_error',  n_jobs=1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs', return_train_score=True, return_estimator=True , error_score='raise')
        for phase in ['train', 'test']:
            scores = all_scores[phase + '_score']
            mean = scores.mean()
            std = scores.std()
            logging.info('%s: mean %s, std=%s, scores=%s', phase, mean, std, scores)

        objective_value = - mean # from test scores

        #print('MY', all_scores)
        assert cross_validation == len(all_scores['estimator'])

        regressors = all_scores['estimator']
        mse_mean_train, result_train = calculate_mean_prediction(regressors, x_train, y_train)
        logging.info('mean mse train=%s, ensemble train result=%s', mse_mean_train, result_train)
        mse_mean_test, result_test = calculate_mean_prediction(regressors, x_test, y_test)
        logging.info('mean mse test=%s, ensemble test result=%s', mse_mean_test, result_test)

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

        calculate_metrics(model, x_train, y_train, metric, 'train')
        objective_value = 0.
        if x_val:
            result = calculate_metrics(model, x_val, y_val, metric, 'val')
            objective_value = result[metric]
        if x_test:
            result = calculate_metrics(model, x_test, y_test, metric, 'test')

    return objective_value

def calculate_metrics(model, x, y, metric, ml_phase):
    y_pred = model.predict(x)
    if metric == 'mse':
        result = oscml.utils.util.calculate_metrics(y, y_pred)
    else: # accuracy
        result = {'accuracy': sklearn.metrics.accuracy_score(y, y_pred)}
        
    logging.info('%s result=%s', ml_phase, result)
    return result
