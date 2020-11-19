import logging
import optuna

# sets model parameter that can have mulitple values (list)
# using config file
def set_config_param_list(trial, param_name, param, length):
    param_suggestions = []
    for i in range(length):
        suggested_value = set_config_param(trial=trial,param_name=param_name+'_{}'.format(i),param=param)
        suggested_value = process_list_param(suggested_value,i)
        param_suggestions.append(suggested_value)
        param = apply_direction(param,suggested_value)
    return param_suggestions

# sets a single model parameter value from the config file
def set_config_param(trial, param_name, param):
    if isinstance(param, dict):
        try:
            param_local = param.copy()
            param_local.pop('direction', None) # remove "direction" if exists
            param_type = param_local.pop('type')
            if param_type == 'categorical':
                # name, choices
                return trial.suggest_categorical(name=param_name, **param_local)
            elif param_type == 'discrete_uniform':
                # name, low, high, q
                return trial.suggest_discrete_uniform(name=param_name, **param_local)
            elif param_type == 'float':
                # name, low, high, *[, step, log]
                return trial.suggest_float(name=param_name, **param_local)
            elif param_type == 'int':
                # name, low, high[, step, log]
                return trial.suggest_int(name=param_name, **param_local)
            elif param_type == 'loguniform':
                # name, low, high
                return trial.suggest_loguniform(name=param_name, **param_local)
            elif param_type == 'uniform':
                # name, low, high
                return trial.suggest_uniform(name=param_name, **param_local)
        except KeyError as exc:
            logging.exception('', exc_info=True)
            raise exc
        except TypeError as exc:
            logging.exception('', exc_info=True)
            raise exc
        except ValueError as exc:
            logging.exception('', exc_info=True)
            raise exc
    else:
        return param

def process_list_param(param,i):
    if isinstance(param,list):
        if i < len(param):
            param = param[i]
        else:
            param = param[-1]
    return param

def apply_direction(param,prev_value):
    if isinstance(param, dict):
        if 'direction' in param:
            if param['direction']=="decreasing":
                param['high'] = prev_value
            elif param['direction']=="increasing":
                param['low'] = prev_value
            elif param['direction']=="constant":
                param = prev_value
    return param