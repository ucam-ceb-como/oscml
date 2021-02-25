import argparse
import datetime
import functools
import json
import logging
import os
import random
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch

import oscml.hpo.objective
import oscml.hpo.optunawrapper


def get_dataframes(dataset, seed):
    df_train, df_val, df_test, transformer = oscml.data.dataset.get_dataframes(dataset=dataset, seed = seed)
    return df_train, df_val, df_test, transformer

def none_or_str(value):
    if value == 'None':
        return None
    return value

def bool_or_str(value):
    if value == 'False':
        return False
    elif value == 'True':
        return True
    return value

# main starting routine for hpo and model training
def start(config_dev=None):

    print('current working directory=', os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='.')
    parser.add_argument('--dst', type=str, default='.')
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=None, help='Stop study after the given number of second(s). If this argument is not set, the study is executed without time limitation.')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--storage', type=none_or_str, default=None)
    parser.add_argument('--study_name', type=none_or_str, default=None)
    parser.add_argument('--load_if_exists', type=bool_or_str, default=False)
    args = parser.parse_args()

    if args.config:
        with open(args.config) as json_config:
            config = json.load(json_config, object_pairs_hook=OrderedDict)
    else:
        config = config_dev

    # seed everything adn choose between deterministic and non-deterministic run
    #--------------------------------------------------------------------------------
    seed = config['numerical_settings'].get('seed')
    cudnn_deterministic = config['numerical_settings'].get('cudnn_deterministic',True)
    cudnn_benchmark = config['numerical_settings'].get('cudnn_benchmark',False)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #--------------------------------------------------------------------------------

    # init file logging
    log_config_file = args.src + '/conf/logging.yaml'
    date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = args.dst + '/logs/hpo_' + date
    oscml.utils.util.init_file_logging(log_config_file, log_dir + '/oscml.log')

    logging.info('current working directory=%s', os.getcwd())
    logging.info('args=%s', args)
    logging.info('config=%s', config)

    df_train, df_val, df_test, transformer = get_dataframes(config['dataset'], seed)

    # concatenate the train and validation dataset to one dataset when cross-validation is on
    cv = config['training']['cross_validation']
    if isinstance(cv, int) and cv > 1:
        df_train = pd.concat([df_train, df_val])
        df_val = None

    n_previous_trials = oscml.hpo.optunawrapper.check_for_existing_study(config['training'].get('storage',args.storage), config['training'].get('study_name',args.study_name))
    n_trials = config['training'].get('n_trials',args.trials)
    if n_previous_trials > 0:
        total_number_trials = n_trials + n_previous_trials - 1
    else:
        total_number_trials = n_trials

    obj = functools.partial(oscml.hpo.objective.objective, config=config,
        df_train=df_train, df_val=df_val, df_test=df_test, transformer=transformer, log_dir=log_dir,
        total_number_trials=total_number_trials)

    return oscml.hpo.optunawrapper.start_hpo(args=args, objective=obj, log_dir=log_dir, config=config, total_number_trials=total_number_trials)


if __name__ == '__main__':
    start()
