import sklearn

import oscml.utils.util
from oscml.utils.util import log

def train_and_test_with_HPO(x_train, y_train, x_test, y_test, model, 
                            params_grid, cross_validation, scoring='neg_mean_squared_error'):
       
    log('fitting and hyperparameter tuning for', model)
    log('sample size =', len(x_train), len(y_train), len(x_test), len(y_test) )
    log('cross_validation=', cross_validation)
    log('params grid=', params_grid)
    
    search = sklearn.model_selection.GridSearchCV(model, params_grid, scoring=scoring, 
                          cv=cross_validation, return_train_score=True)
    
    search.fit(x_train, y_train)
    
    y_pred_train = search.predict(x_train)
    y_pred_test = search.predict(x_test)
    
    if scoring == 'neg_mean_squared_error':
        result_train = oscml.utils.util.calculate_metrics(y_train, y_pred_train)
        result_test = oscml.utils.util.calculate_metrics(y_test, y_pred_test)
    else: # accuracy
        result_train = {'accuracy': sklearn.metrics.accuracy_score(y_train, y_pred_train)}
        result_test = {'accuracy': sklearn.metrics.accuracy_score(y_test, y_pred_test)}
        
    log('best params=', search.best_params_)
    log('best mean test score=', search.best_score_)
    log('train result=', result_train)
    log('test result=', result_test)
        
    return result_train, result_test, search