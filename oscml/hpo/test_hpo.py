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
            "src": "./data/processed/HOPV_15_revised_2_processed_homo.csv",
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
            "metric": 'val_loss',
            "direction": 'minimize',
            "cross_validation": False
        }

    numerical_settings ={
	        "seed": 1,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False
        }

    return {
        "numerical_settings": numerical_settings,
        "dataset": dataset,
        "model":{
            "name": model_name,
            "model_specific": model_specific,            
        },
        "training": training
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
        oscml.utils.util.init_logging('.', './tmp')

    def test_train_gnn_cep25000_one_trial(self):

        config = create_config_simplegnn(oscml.data.dataset_cep.CEP25000)

        testargs = ['test',
            '--trials', '1',
        ]
        with unittest.mock.patch('sys.argv', testargs):
            oscml.hpo.train.start(config_dev=config)

    def test_train_gnn_hopv15_one_trial(self):

        config = create_config_simplegnn(oscml.data.dataset_hopv15.HOPV15)

        testargs = ['test',
            '--trials', '1',
        ]
        with unittest.mock.patch('sys.argv', testargs):
            oscml.hpo.train.start(config_dev=config)

    def test_train_bilstm_cep25000_one_trial(self):

        config = create_config_bilstm(oscml.data.dataset_cep.CEP25000)

        testargs = ['test',
            '--trials', '1',
        ]
        with unittest.mock.patch('sys.argv', testargs):
            oscml.hpo.train.start(config_dev=config)

    def test_train_bilstm_hopv15_one_trial(self):

        config = create_config_bilstm(oscml.data.dataset_hopv15.HOPV15)

        testargs = ['test',
            '--trials', '1',
        ]
        with unittest.mock.patch('sys.argv', testargs):
            oscml.hpo.train.start(config_dev=config)
    
    def test_train_attentiveFP_cep25000_simple_featurizer(self):

        config = create_config_attentivefp(oscml.data.dataset_cep.CEP25000, 'simple')

        testargs = ['test',
            '--trials', '1',
            #'--dataset', oscml.data.dataset_cep.CEP25000,
            ]
        with unittest.mock.patch('sys.argv', testargs):
            oscml.hpo.train.start(config_dev=config)

    def test_hpo_attentiveFP_hopv15_full_featurizer(self):

        config = create_config_attentivefp(oscml.data.dataset_hopv15.HOPV15, 'full')

        testargs = ['test',
            '--trials', '2',
            #'--dataset', oscml.data.dataset_hopv15.HOPV15,
            ]
        with unittest.mock.patch('sys.argv', testargs):
            oscml.hpo.train.start(config_dev=config)

    def test_hpo_attentiveFP_hopv15_full_featurizer_with_config_file(self):

        testargs = ['test',
            '--trials', '2',
            '--config', './res/test_confhpo/confhpo_attentivefp_hopv15.json',
            ]
        with unittest.mock.patch('sys.argv', testargs):
            oscml.hpo.train.start()

    def test_load_model_from_checkpoint(self):

        #load model
        path = './res/test_checkpoint_gnn_cep25000_e10/last.ckpt'
        model_class = oscml.models.model_gnn.SimpleGNN
        model = model_class.load_from_checkpoint(path)

        logging.info(model)

        print(model.optimizer)

        self.assertEqual('SGD', model.optimizer['name'])
        self.assertEqual(4, len(model.conv_modules))
        # 2 hidden layer with 2 ReLUs and Dropout and 1 output layer
        self.assertEqual(7, len(model.mlp))
        # check second layer
        self.assertEqual(11, model.mlp[0].in_features)
        self.assertEqual(11, model.mlp[0].out_features)

        self.assertAlmostEqual(4.126659584567247, model.target_mean, 4)
        self.assertAlmostEqual(2.407382175602577, model.target_std, 4)

        # get dataloaders
        path = oscml.data.dataset.path_cepdb_25000('.')
        df_train, df_val, df_test = oscml.data.dataset.read_and_split(path)
        transformer = oscml.data.dataset.DataTransformer('pce', model.target_mean, model.target_std, 'SMILES_str')
        #train_dl, val_dl, test_dl = oscml.models.model_gnn.get_dataloaders_CEP(df_train, df_val, df_test, 250, transformer)
        train_dl, val_dl, test_dl = oscml.models.model_gnn.get_dataloaders(oscml.data.dataset_cep.CEP25000, df_train, df_val, df_test, transformer, 250)

        # prepare Lightning trainer
        params = oscml.utils.util_lightning.get_standard_params_for_trainer('val_loss') 
        params.update({
            'max_epochs': 1,
            'logger': pl.loggers.CSVLogger(save_dir='./tmp/logs', name='resumed') ,
        })

        trainer = pl.Trainer(**params)
        # doesn't work correctly:
        #trainer = pl.Trainer(resume_from_checkpoint=path)
    
        # test
        result = trainer.test(model, test_dataloaders=val_dl) 
        print(result)
        # the value is the validation error at epoch 10 copied from the corresponding log file / metric.csv file
        self.assertAlmostEqual(5.795882524953438, result[0]['mse'], 4)


    def test_gnn_cep25000_ckpt_test_only(self):
        testargs = ['test', 
            '--epochs', '0',
            '--ckpt',  './res/test_checkpoint_gnn_cep25000_e10/last.ckpt',
            #'--dataset', oscml.data.dataset_cep.CEP25000,
            '--model', 'SimpleGNN',
        ]

        with unittest.mock.patch('sys.argv', testargs):
            result = oscml.hpo.resume.start()
            # the value is the test error at epoch 10 copied from the corresponding log file / metric.csv file           
            self.assertAlmostEqual(5.80646424256214, result['mse'], 4)
    
    def test_gnn_cep25000_ckpt_resume_training(self):
        testargs = ['test', 
            '--epochs', '1',
            '--ckpt',  './res/test_checkpoint_gnn_cep25000_e10/last.ckpt',
            #'--dataset', oscml.data.dataset_cep.CEP25000,
            '--model', 'SimpleGNN',
        ]

        with unittest.mock.patch('sys.argv', testargs):
            result = oscml.hpo.resume.start()
            logging.info('result=%s', result)

    def test_train_rf_cep25000_one_trial_with_config_file(self):

        testargs = ['test',
            '--trials', '1',
            '--config', './res/test_confhpo/confhpo_rf_cep25000.json',
        ]
        with unittest.mock.patch('sys.argv', testargs):
            oscml.hpo.train.start()

    def test_train_svr_hopv15_one_trial_with_config_file(self):

        testargs = ['test',
            '--trials', '1',
            '--config', './res/test_confhpo/confhpo_svr_hopv15.json',
        ]
        with unittest.mock.patch('sys.argv', testargs):
            oscml.hpo.train.start()


if __name__ == '__main__':

    #unittest.main()

    suite = unittest.TestSuite()
    suite.addTest(Test_HPO('test_train_gnn_cep25000_one_trial'))
    suite.addTest(Test_HPO('test_train_gnn_hopv15_one_trial'))
    suite.addTest(Test_HPO('test_train_bilstm_cep25000_one_trial'))
    suite.addTest(Test_HPO('test_train_bilstm_hopv15_one_trial'))
    suite.addTest(Test_HPO('test_train_attentiveFP_cep25000_simple_featurizer'))
    suite.addTest(Test_HPO('test_hpo_attentiveFP_hopv15_full_featurizer'))
    suite.addTest(Test_HPO('test_hpo_attentiveFP_hopv15_full_featurizer_with_config_file'))
    #suite.addTest(Test_HPO('test_load_model_from_checkpoint'))
    #suite.addTest(Test_HPO('test_gnn_cep25000_ckpt_test_only'))
    #suite.addTest(Test_HPO('test_gnn_cep25000_ckpt_resume_training'))
    suite.addTest(Test_HPO('test_train_rf_cep25000_one_trial_with_config_file'))
    suite.addTest(Test_HPO('test_train_svr_hopv15_one_trial_with_config_file'))
    runner = unittest.TextTestRunner()
    runner.run(suite)
