import logging
import os
import time
import unittest
from unittest.mock import patch

import pytorch_lightning as pl

import oscml.data.dataset_cep
import oscml.data.dataset_hopv15
import oscml.hpo.optunawrapper
import oscml.hpo.resume
import oscml.hpo.train
import oscml.utils.util


def create_config(type_dict, model_name, model_specific):

    if type_dict == oscml.data.dataset_cep.CEP25000:
        dataset = {
            "src": "./data/processed/CEPDB_25000.csv",
            "z-stand": "True",
            "x_column": ["SMILES_str"],
            "y_column": ["pce"],
            "split": "ml_phase"
        }
    elif type_dict == oscml.data.dataset_hopv15.HOPV15:
        dataset = {
            "src": "./data/processed/HOPV_15_revised_2_processed_homo_5fold.csv",
            "z-stand": "True",
            "x_column": ["smiles"],
            "y_column": ["pce"],
            "split": [200, None, 36]
        }

    training = {
            "optimiser":{
                "name":"Adam",
                "lr":0.001,
                "weight_decay":0.0,
                },
            "batch_size": 250,
            "epochs": 1,
            "patience": -1,
            "min_delta": 1,
            "metric": 'val_loss',
            "direction": 'minimize',
            "cross_validation": False
        }

    numerical_settings ={
	        "seed": 1,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False
        }

    post_processing ={
        "contour_plot": False,
        "best_trial_retraining": False,
        "z_transform_inverse_prediction": True,
        "regression_plot": False
    }

    return {
        "numerical_settings": numerical_settings,
        "dataset": dataset,
        "model":{
            "name": model_name,
            "model_specific": model_specific,
        },
        "training": training,
        "post_processing": post_processing
    }

def create_config_attentivefp(type_dict, featurizer):
    model_specific = {
        'graph_feat_size': 200,
        'num_layers': 4,
        'num_timesteps': 2,
        'dropout': 0.,
    }
    d = create_config(type_dict, 'AttentiveFP', model_specific)
    d['model'].update({'featurizer': featurizer})
    return d

def create_config_bilstm(type_dict):
    model_specific = {
        'embedding_dim': 128,
        'mlp_layers': 3,
        'mlp_units': [64, 32, 16],
        "mlp_dropouts": [0., 0., 0.]
    }
    d = create_config(type_dict, 'BILSTM', model_specific)
    if type_dict == oscml.data.dataset_cep.CEP25000:
        max_sequence_length = 60
    else:
        max_sequence_length = 150
    d['model'].update({'type_dict': type_dict, 'max_sequence_length': max_sequence_length})
    return d

def create_config_simplegnn(type_dict):
    model_specific = {
        'embedding_dim': 128,
        "conv_layers": 4,
        "conv_dims": [128, 128, 128, 128],
        "mlp_layers": 2,
        "mlp_units": [64, 32],
        "mlp_dropouts": [0.1, 0.2],
    }
    d = create_config(type_dict, 'SimpleGNN', model_specific)
    d['model'].update({'type_dict': type_dict})
    return d

class Test_HPO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print()
        print()
        print('###################################')
        print('#             HPO Tests           #')
        print('###################################')
        print()
        print()

    def test_train_gnn_cep25000_one_trial(self):
        print()
        print()
        print('--------------------------------------------------')
        print('-     Test: test_train_gnn_cep25000_one_trial    -')
        print('--------------------------------------------------')
        print()
        print()
        config = create_config_simplegnn(oscml.data.dataset_cep.CEP25000)

        testargs = ['test',
            '--trials', '1',
            '--src', './tests',
            '--dst', './tests/tests_logs'
        ]
        with unittest.mock.patch('sys.argv', testargs):
            oscml.hpo.train.start(config_dev=config)

    def test_train_gnn_hopv15_one_trial(self):
        print()
        print()
        print('--------------------------------------------------')
        print('-     Test: test_train_gnn_hopv15_one_trial      -')
        print('--------------------------------------------------')
        print()
        print()
        config = create_config_simplegnn(oscml.data.dataset_hopv15.HOPV15)

        testargs = ['test',
            '--trials', '1',
            '--src', './tests',
            '--dst', './tests/tests_logs'
        ]
        with unittest.mock.patch('sys.argv', testargs):
            oscml.hpo.train.start(config_dev=config)

    def test_train_bilstm_cep25000_one_trial(self):
        print()
        print()
        print('--------------------------------------------------')
        print('-   Test: test_train_bilstm_cep25000_one_trial   -')
        print('--------------------------------------------------')
        print()
        print()
        config = create_config_bilstm(oscml.data.dataset_cep.CEP25000)

        testargs = ['test',
            '--trials', '1',
            '--src', './tests',
            '--dst', './tests/tests_logs'
        ]
        with unittest.mock.patch('sys.argv', testargs):
            oscml.hpo.train.start(config_dev=config)

    def test_train_bilstm_hopv15_one_trial(self):
        print()
        print()
        print('--------------------------------------------------')
        print('-   Test: test_train_bilstm_hopv15_one_trial     -')
        print('--------------------------------------------------')
        print()
        print()
        config = create_config_bilstm(oscml.data.dataset_hopv15.HOPV15)

        testargs = ['test',
            '--trials', '1',
            '--src', './tests',
            '--dst', './tests/tests_logs'
        ]
        with unittest.mock.patch('sys.argv', testargs):
            oscml.hpo.train.start(config_dev=config)

    def test_train_attentiveFP_cep25000_simple_featurizer(self):
        print()
        print()
        print('-------------------------------------------------------------')
        print('-  Test: test_train_attentiveFP_cep25000_simple_featurizer  -')
        print('-------------------------------------------------------------')
        print()
        print()
        config = create_config_attentivefp(oscml.data.dataset_cep.CEP25000, 'simple')

        testargs = ['test',
            '--trials', '1',
            '--src', './tests',
            '--dst', './tests/tests_logs'
            #'--dataset', oscml.data.dataset_cep.CEP25000,
            ]
        with unittest.mock.patch('sys.argv', testargs):
            oscml.hpo.train.start(config_dev=config)

    def test_hpo_attentiveFP_hopv15_full_featurizer(self):
        print()
        print()
        print('-------------------------------------------------------------')
        print('-     Test: test_hpo_attentiveFP_hopv15_full_featurizer     -')
        print('-------------------------------------------------------------')
        print()
        print()
        config = create_config_attentivefp(oscml.data.dataset_hopv15.HOPV15, 'full')

        testargs = ['test',
            '--trials', '2',
            '--src', './tests',
            '--dst', './tests/tests_logs'
            #'--dataset', oscml.data.dataset_hopv15.HOPV15,
            ]
        with unittest.mock.patch('sys.argv', testargs):
            oscml.hpo.train.start(config_dev=config)

    def test_hpo_attentiveFP_hopv15_full_featurizer_with_config_file(self):
        print()
        print()
        print('------------------------------------------------------------------------')
        print('-  Test: test_hpo_attentiveFP_hopv15_full_featurizer_with_config_file  -')
        print('------------------------------------------------------------------------')
        print()
        print()
        testargs = ['test',
            '--trials', '2',
            '--config', './tests/test_confhpo/confhpo_attentivefp_hopv15.json',
            '--src', './tests',
            '--dst', './tests/tests_logs'
            ]
        with unittest.mock.patch('sys.argv', testargs):
            oscml.hpo.train.start()

    def test_train_rf_cep25000_one_trial_with_config_file(self):
        print('Test: test_train_rf_cep25000_one_trial_with_config_file')
        print()
        testargs = ['test',
            '--trials', '1',
            '--config', './tests/test_confhpo/confhpo_rf_cep25000.json',
            '--src', './tests',
            '--dst', './tests/tests_logs'
        ]
        with unittest.mock.patch('sys.argv', testargs):
            oscml.hpo.train.start()

    def test_train_svr_hopv15_one_trial_with_config_file(self):
        print()
        print()
        print('---------------------------------------------------------------------')
        print('-       Test: test_train_svr_hopv15_one_trial_with_config_file      -')
        print('---------------------------------------------------------------------')
        print()
        print()
        testargs = ['test',
            '--trials', '1',
            '--config', './tests/test_confhpo/confhpo_svr_hopv15.json',
            '--src', './tests',
            '--dst', './tests/tests_logs'
        ]
        with unittest.mock.patch('sys.argv', testargs):
            oscml.hpo.train.start()
