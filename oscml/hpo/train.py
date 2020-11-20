import argparse
import datetime
import functools
import logging
import os
import json

import oscml.hpo.objective
import oscml.hpo.optunawrapper


def get_dataframes(dataset):
    df_train, df_val, df_test, transformer = oscml.data.dataset.get_dataframes(dataset=dataset, train_size=283, test_size=30)
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
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--trials', type=int, default=None)
    parser.add_argument('--timeout', type=int, default=None, help='Stop study after the given number of second(s). If this argument is not set, the study is executed without time limitation.')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--config', type=str, default=None)
    #parser.add_argument('--model', type=str, default=None, choices=['BILSTM', 'AttentiveFP', 'SimpleGNN'])
    #parser.add_argument('--ckpt', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--datasetpath', type=str, default=None)
    parser.add_argument('--seed', type=int, default=200)
    parser.add_argument('--cv', type=int, default=None)
    parser.add_argument('--storage', type=none_or_str, default=None)
    parser.add_argument('--study_name', type=none_or_str, default=None)
    parser.add_argument('--load_if_exists', type=bool_or_str, default=False)
    parser.add_argument('--metric', type=str, default='val_loss')
    parser.add_argument('--direction', type=str, default='minimize')
    parser.add_argument('--featurizer', type=str, choices=['simple', 'full'], default='full')
    args = parser.parse_args()

    # init file logging
    log_config_file = args.src + '/conf/logging.yaml'
    date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = args.dst + '/logs/hpo_' + date
    oscml.utils.util.init_file_logging(log_config_file, log_dir + '/oscml.log')

    logging.info('current working directory=%s', os.getcwd())
    args.log_dir = log_dir
    if args.datasetpath:
        args.datasetpath = args.src + '/' + args.datasetpath
    logging.info('args=%s', args)

    if args.config:
        with open(args.config) as json_config:
            config = json.load(json_config)
    else:
        config = config_dev

    # temporary copy for refactoring towards dataset section in configuration file
    if 'type_dict' in config['model']:
        config['dataset'].update({'type_dict' : config['model']['type_dict']})

    logging.info('config=%s', config)

    df_train, df_val, df_test, transformer = get_dataframes(config['dataset'])

    obj = functools.partial(oscml.hpo.objective.objective, config=config, args=args,
        df_train=df_train, df_val=df_val, df_test=df_test, transformer=transformer)

    return oscml.hpo.optunawrapper.start_hpo(args=args, objective=obj)


if __name__ == '__main__':
    start()
