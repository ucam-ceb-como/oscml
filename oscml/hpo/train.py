import argparse
import datetime
import functools
import logging
import os

import oscml.hpo.objective
import oscml.hpo.optunawrapper


def get_dataframes(src, dataset):
    df_train, df_val, df_test, transformer = oscml.data.dataset.get_dataframes(dataset=dataset, src=src, train_size=283, test_size=30)
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
    parser.add_argument('--model', type=str, default=None, choices=['BILSTM', 'AttentiveFP', 'SimpleGNN'])
    parser.add_argument('--fixedtrial', type=bool, default=False)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--dataset', type=str)
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


    if args.config:
        # TODO: read the config file given by args.config which contains model etc.
        config = {'model': args.model}
    else:
        config = {'model': config_dev['model_name']}
        config.update(config_dev["model"])
        config.update(config_dev["optimizer"])

    logging.info('config file=%s', config)

    df_train, df_val, df_test, transformer  = get_dataframes(args.src, args.dataset)

    obj = functools.partial(oscml.hpo.objective.objective, config=config, 
        df_train=df_train, df_val=df_val, df_test=df_test, transformer=transformer, log_dir=log_dir, 
        featurizer=args.featurizer)

    return oscml.hpo.optunawrapper.start_hpo(
            args=args,
            objective=obj,
            log_dir=log_dir,
            fixed_trial_params=config)


if __name__ == '__main__':
    start()
