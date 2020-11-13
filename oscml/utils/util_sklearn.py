import logging

import sklearn

import oscml.utils.util
from oscml.utils.util import concat

def train_and_test(x_train, y_train, x_test, y_test, model, cross_validation, metric):
       
    logging.info(concat('fitting for', model))
    len_x_test = (len(x_test)) if x_test else None
    len_y_test = (len(y_test)) if y_test else None
    logging.info(concat('sample size =', len(x_train), len(y_train), len_x_test, len_y_test))
    logging.info(concat('cross_validation=', cross_validation))
    
    if cross_validation:
        # use cross_validate instead of cross_val_score to get more information about scores
        # only use 1 CPU (n_jobs=1)
        all_scores = sklearn.model_selection.cross_validate(model, x_train, y_train, cv=cross_validation,
                    scoring='neg_mean_squared_error',  n_jobs=1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs', return_train_score=True, return_estimator=True , error_score='raise')
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
    else:

        model.fit(x_train, y_train)
    
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)
    
        if metric == 'mse':
            result_train = oscml.utils.util.calculate_metrics(y_train, y_pred_train)
            result_test = oscml.utils.util.calculate_metrics(y_test, y_pred_test)
        else: # accuracy
            result_train = {'accuracy': sklearn.metrics.accuracy_score(y_train, y_pred_train)}
            result_test = {'accuracy': sklearn.metrics.accuracy_score(y_test, y_pred_test)}
            
        logging.info(concat('train result=', result_train))
        logging.info(concat('test result=', result_test))
        
        objective_value = result_test[metric]

    return objective_value