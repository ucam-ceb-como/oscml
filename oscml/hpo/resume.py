import argparse
import datetime
import logging
import os

import oscml.data.dataset
import oscml.hpo.objective
import oscml.hpo.train
import oscml.models.model_bilstm
import oscml.models.model_gnn


def resume(args):

    df_train, df_val, df_test, transformer = oscml.hpo.train.get_dataframes(args.src, args.dataset, args.datasetpath)

    if args.model == 'SimpleGNN':
        train_dl, val_dl, test_dl = oscml.models.model_gnn.get_dataloaders(args.dataset, df_train, df_val, df_test,
            transformer, batch_size=250)
        model_class = oscml.models.model_gnn.SimpleGNN

    #elif args.model == 'BILSTM':
    #    model_class = oscml.models.model_bilstm.BiLstmForPce

    model = model_class.load_from_checkpoint(args.ckpt)

    if args.epochs > 0:
        trainer_params = {}
        # n_trials = 1 to activate saving checkpoints
        result = oscml.hpo.objective.fit_or_test(model, train_dl, val_dl, test_dl, trainer_params,
                    args.epochs, args.metric, args.log_dir, n_trials=1)

    else:
        trainer_params = {}
        result = oscml.hpo.objective.fit_or_test(model, None, None, test_dl, trainer_params,
                    args.epochs, args.metric, args.log_dir)
    
    logging.info('result=%s', result)
    return result


def start():

    print('current working directory=', os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='.')
    parser.add_argument('--dst', type=str, default='.')
    parser.add_argument('--epochs', type=int, default=0)
    #parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--model', type=str, default=None, choices=['SimpleGNN']) #['BILSTM', 'AttentiveFP', 'SimpleGNN'])
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--datasetpath', type=str, default=None)
    parser.add_argument('--metric', type=str, default='val_loss')
    #parser.add_argument('--seed', type=int, default=200)
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

    return resume(args)


if __name__ == '__main__':
    start()