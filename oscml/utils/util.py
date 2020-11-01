import datetime
import logging
import logging.config
import os
import pprint
import pytorch_lightning as pl
from time import sleep
import yaml

import numpy as np
import rdkit.Chem
import sklearn.metrics
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import oscml.utils.params
from oscml.utils.params import cfg

def init_logging(log_config_file, log_file):
    print('initializing logging with log config file=', log_config_file, ', log file=', log_file)
    with open(log_config_file, 'r') as f:
        # always use safe_load to avoid reading and executing as YAML serialized Python code
        # yaml returns a dictionary
        log_cfg = yaml.safe_load(f.read())

    pprint_log_cfg = pprint.PrettyPrinter().pformat(log_cfg)
    print(pprint_log_cfg)

    if log_file:
        log_cfg['handlers']['file_handler']['filename'] = log_file
    else:
        log_file = log_cfg['handlers']['file_handler']['filename']

    dir_name = os.path.dirname(log_file)
    try:
        os.makedirs(dir_name, exist_ok=True)
    except FileExistsError:
        print('dir existiert bereits, dir=', dir_name)

    # use logging configuration with dictionary
    logging.config.dictConfig(log_cfg)

    log('initialized logging with config file=', log_config_file, ', log file=', log_file)

def init_standard_logging():

    # file and console logging with Python's standard logging library
    log_config_file = './conf/logging.yaml'
    log_dir = './tmp/log_tmp'
    name = 'test_train_model_without_hpo'
    oscml.utils.util.init_logging(log_config_file, log_dir + '/' + name + '.log')

    # metric logging with PyTorch Lightning
    return pl.loggers.CSVLogger(save_dir=log_dir, name=name, version=None)

def log(*args):
    if len(args) == 1:
        logging.getLogger().info(args[0])
    else:
        message = ''
        for m in args:
            message += str(m) + ' '
        logging.getLogger().info(message)
       
def logm(*args):
    logging.getLogger().info(args)

def smiles2mol(smiles):
    m = rdkit.Chem.MolFromSmiles(smiles)
    if m and cfg[oscml.utils.params.INCLUDE_HYDROGENS]:
        m = rdkit.Chem.AddHs(m)
    return m

def smiles2mol_df(df, column):
    log('generating RDKit molecules, column=', column)
    sleep(1)
    x = []
    for i in tqdm(range(len(df))):
        smiles = df.iloc[i][column]
        m = smiles2mol(smiles)
        x.append(m)
    sleep(1)
    return x

def mol_with_atom_index(mol):
    """Use this method to add RDKit's atom index (order) to the visualized graph"""
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol


def calculate_metrics(y_true_np, y_pred_np):
    """
    mean squared error: $mse = \frac{1}{n} \sum_i ( y_i - \hat{y}_i )^2$

    coefficient of determination: $R^2 = 1 - \text{residual sum of squares} \div \text{total sum of squares} = 1 - \sum_i (y_i - \hat{y}_i)^2 \div \sum_i (y_i - \bar{y})^2$

    Pearson correlation coefficient: $r = \text{cov}(Y, \hat{Y})  \div \sigma(Y) \space \sigma(\hat{Y}) = \sum_i (y_i -\bar{y}) (\hat{y}_i - \bar{\hat{y}}) \div \sqrt{\sum_i (y_i - \bar{y})^2} \sqrt{\sum_i (\hat{y}_i - \bar{\hat{y}})^2}$

    This method calculates all three metrics for two numpy arrays of $y_1,\dots,y_n$ and $\hat{y}_1,\dots,\hat{y}_n$.

    When using normalized values such as 'pcez' instead of 'pce' for training, we have to transform back the predicted values (on the validation and training set) before calling calculate_metrics.
    """
    mae = sklearn.metrics.mean_absolute_error(y_true_np, y_pred_np)
    # mse is the mean squared error because squared=True by default
    mse = sklearn.metrics.mean_squared_error(y_true_np, y_pred_np, squared=True)
    rmse = sklearn.metrics.mean_squared_error(y_true_np, y_pred_np, squared=False)
    R2 = sklearn.metrics.r2_score(y_true_np, y_pred_np)
    r_Pearson = np.corrcoef(y_true_np, y_pred_np)
    return {'mse':mse, 'rmse': rmse, 'R2':R2, 'r':r_Pearson[0,1], 'mae': mae, 'count': len(y_true_np)}

def create_tensorboard_logger(postfix=''):
    log_dir = cfg[oscml.utils.params.TENSOR_BOARD_MAIN_LOG_DIR] + '/' + datetime.datetime.now().strftime('%y%m%d_%H%M')
    log_dir_val = log_dir + '_val_' + postfix
    val_tb = SummaryWriter(log_dir=log_dir_val, flush_secs=10)
    log_dir_train = log_dir + '_train_' + postfix
    train_tb = SummaryWriter(log_dir=log_dir_train, flush_secs=10)
    return (train_tb, val_tb)