import argparse
import datetime
import functools
import logging
import os
import json
from collections import OrderedDict

import torch
import numpy as np
import random

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

def start(config_dev=None):

    print('current working directory=', os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='.')
    parser.add_argument('--dst', type=str, default='.')
    #parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--trials', type=int, default=None)
    parser.add_argument('--timeout', type=int, default=None, help='Stop study after the given number of second(s). If this argument is not set, the study is executed without time limitation.')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--config', type=str, default=None)
    #parser.add_argument('--model', type=str, default=None, choices=['BILSTM', 'AttentiveFP', 'SimpleGNN'])
    #parser.add_argument('--ckpt', type=str)
    #parser.add_argument('--dataset', type=str)
    #parser.add_argument('--datasetpath', type=str, default=None)
    parser.add_argument('--seed', type=int, default=200)
    parser.add_argument('--cv', type=int, default=None)
    parser.add_argument('--storage', type=none_or_str, default=None)
    parser.add_argument('--study_name', type=none_or_str, default=None)
    parser.add_argument('--load_if_exists', type=bool_or_str, default=False)
    #parser.add_argument('--metric', type=str, default='val_loss')
    # TODO default for --direction is quite error prone for HPO, can easily be forgotten
    # should it be moved to the config file?
    # should we provide a HPO section in the config file?
    parser.add_argument('--direction', type=str, default='minimize', choices=['minimize', 'maximize'])
    #parser.add_argument('--featurizer', type=str, choices=['simple', 'full'], default='full')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    # init file logging
    log_config_file = args.src + '/conf/logging.yaml'
    date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = args.dst + '/logs/hpo_' + date
    oscml.utils.util.init_file_logging(log_config_file, log_dir + '/oscml.log')

    logging.info('current working directory=%s', os.getcwd())
    logging.info('args=%s', args)

    if args.config:
        with open(args.config) as json_config:
            config = json.load(json_config, object_pairs_hook=OrderedDict)
    else:
        config = config_dev

    logging.info('config=%s', config)

    df_train, df_val, df_test, transformer = get_dataframes(config['dataset'], args.seed)

    obj = functools.partial(oscml.hpo.objective.objective, config=config, args=args,
        df_train=df_train, df_val=df_val, df_test=df_test, transformer=transformer, log_dir=log_dir)

    return oscml.hpo.optunawrapper.start_hpo(args=args, objective=obj, log_dir=log_dir)


if __name__ == '__main__':
    start()
