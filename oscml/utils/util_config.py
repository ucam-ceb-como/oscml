import logging
import optuna

# returns value(s) of a parameter defined in the config file
def set_config_param(trial, param_name, param, all_params):
    # trial      - optuna trial object
    # param_name - base paramter name you wish to use
    # param      - parameter from the config, can be a single number, string, list or dictionary
    # all_params - dictionary of all processed parameters so far

    # if param is a list, process it as list
    if isinstance(param, list):
        param_suggestion = set_config_param_list(trial, param_name, param)
    elif isinstance(param, dict):
        # parameter defined as a dictionary can be used for:
        # 1) defining multiple values of a parameter via list and adding required
        #    list length, e.g.:
        #      - param : {"values": [1,3, 4], "length":5} - this would create [1,3,4,4,4]
        #      - param : {"values": [1,3, 4], "length":"other_param"} - this would create a list whose length
        #            will be equal to the value assigned to the "other_param". "other_param" value must be
        #            a single integer number and must be defined in the config file prior to the current param!
        #   Note:  - param : {"values": [1,3, 4]} - this allowed, but it is better to simply use - param : [1,3, 4]
        #   Note: parameters defined via the above options are NOT ADDED TO TRIAL object
        # 2) defining a single or multiple values of a parameter via optuna sampler, e.g.:
        #   - param : {"optuna_keys:values"} - this would ADD A SINGLE PARAMETER TO TRIAL and sample its value using optuna
        #            suggest function
        #   - param : {"optuna_keys:values","length":5} - this would ADD 5 PARAMETERS TO TRIAL and sample their values using optuna
        #            suggest function
        #   - param : {"optuna_keys:values","length":"other_param"} - this would ADD N PARAMETERS TO TRIAL and sample their values using optuna
        #            suggest function, where N is the value assigned to the "other_param".
        values = param.get('values')
        length = param.get('length')

        if values is not None:
            # this is a list-based parameter
            if length is not None:
                if isinstance(length, str):
                    length = all_params[param['length']]
            param_suggestion = set_config_param_list(trial, param_name, param['values'], length)
        else:
            if length is not None:
                if isinstance(length, str):
                    length = all_params[param['length']]
                param_suggestion = set_config_param_list(trial, param_name, param, length)
            else:
                param_suggestion = set_config_param_single(trial, param_name, param)
    else:
        # this is a single value, fixed parameter:
        # e.g. param : 1 or param : "Adam" etc..
        param_suggestion = set_config_param_single(trial, param_name, param)
    return param_suggestion


# sets model parameter that can have mulitple values (list)
def set_config_param_list(trial, param_name, param, length=None):
    if length is None:
        length = len(param)
    param_suggestions = []
    for i in range(length):
        suggested_value = set_config_param_single(trial=trial,param_name=param_name+'_{}'.format(i),param=param)
        suggested_value = process_list_param(suggested_value,i)
        param_suggestions.append(suggested_value)
        param = apply_direction(param,suggested_value)
    return param_suggestions

# sets a single model parameter value from the config file
def set_config_param_single(trial, param_name, param):
    if isinstance(param, dict):
        if 'type' in param:
            try:
                param_local = param.copy()
                param_local.pop('direction', None) # remove "direction" if exists
                param_local.pop('length', None) # remove "direction" if exists
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