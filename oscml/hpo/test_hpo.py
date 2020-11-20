import logging
import os
import time
import unittest
from unittest.mock import patch

import pytorch_lightning as pl

import oscml.hpo.optunawrapper
import oscml.hpo.resume
import oscml.hpo.train
import oscml.hpo.start_bilstm_with_hpo
import oscml.hpo.start_gnn_with_hpo
import oscml.hpo.start_mnist_with_hpo
import oscml.hpo.start_rf_with_hpo
import oscml.utils.util


def create_config(model_name, model_specific):
    return {
        "model":{
            "name": model_name,
            "model_specific": model_specific,
        },
        "training":{
            "optimiser":{
                "name":"Adam",
                "lr":0.001,
                "weight_decay":0.0,
                },
            "batch_size": 250,
            "epochs": 1,
            "patience": -1,
            "metric": 'val_loss'
        }
    }

def create_config_attentivefp():
    model_specific = {
        'graph_feat_size': 200,
        'num_layers': 4,
        'num_timesteps': 2,
        'dropout': 0.,
    }
    return create_config('AttentiveFP', model_specific)

def create_config_bilstm():
    model_specific = {
        'embedding_dimension': 128,
        'mlp_layers': 3,
        'mlp_units': [64, 32, 16],
        "mlp_dropouts": 0.
    }
    return create_config('BILSTM', model_specific)

def create_config_simplegnn():
    model_specific = {
        'embedding_dimension': 128,
        "conv_layers": 4,
        "conv_dims": [128, 128, 128, 128],
        "mlp_layers": 2,
        "mlp_units": [64, 32],
        "mlp_dropouts": 0.2,
    }
    return create_config('SimpleGNN', model_specific)


class Test_HPO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        oscml.utils.util.init_logging('.', './tmp')

    def test_train_gnn_cep25000_one_trial(self):

        config = create_config_simplegnn()

        testargs = ['test',
            '--dataset', 'CEP25000',
            '--epochs', '1',
            '--trials', '1',
        ]
        with unittest.mock.patch('sys.argv', testargs):
            oscml.hpo.train.start(config_dev=config)

    def test_train_gnn_hopv15_one_trial(self):

        config = create_config_simplegnn()

        testargs = ['test',
            '--dataset', 'HOPV15',
            '--epochs', '1',
            '--trials', '1',
        ]
        with unittest.mock.patch('sys.argv', testargs):
            oscml.hpo.train.start(config_dev=config)

    def test_train_bilstm_cep25000_one_trial(self):

        config = create_config_bilstm()

        testargs = ['test',
            '--dataset', 'CEP25000',
            '--epochs', '1',
            '--trials', '1',
        ]
        with unittest.mock.patch('sys.argv', testargs):
            oscml.hpo.train.start(config_dev=config)

    def test_train_bilstm_hopv15_one_trial(self):

        config = create_config_bilstm()

        testargs = ['test',
            '--dataset', 'HOPV15',
            '--epochs', '2',
            '--trials', '1',
        ]
        with unittest.mock.patch('sys.argv', testargs):
            oscml.hpo.train.start(config_dev=config)
    
    def test_train_attentiveFP_cep25000_simple_featurizer(self):

        config = create_config_attentivefp()

        testargs = ['test',
            '--dataset', 'CEP25000',
            '--epochs', '1',
            '--trials', '1',
            '--featurizer', 'simple'
            ]
        with unittest.mock.patch('sys.argv', testargs):
            oscml.hpo.train.start(config_dev=config)

    def test_hpo_attentiveFP_hopv15_full_featurizer(self):

        config = create_config_attentivefp()

        testargs = ['test',
            '--dataset', 'HOPV15',
            '--epochs', '1',
            '--trials', '2'         # will run twice with the same fixed params
            ]
        with unittest.mock.patch('sys.argv', testargs):
            oscml.hpo.train.start(config_dev=config)

    def test_hpo_attentiveFP_hopv15_full_featurizer_with_config_file(self):

        testargs = ['test',
            '--dataset', 'HOPV15',
            '--epochs', '1',
            '--trials', '2',
            '--config', './conf/confhpo_attentivefp.json'
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
            '--dataset', oscml.data.dataset_cep.CEP25000,
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
            '--dataset', oscml.data.dataset_cep.CEP25000,
            '--model', 'SimpleGNN',
        ]

        with unittest.mock.patch('sys.argv', testargs):
            result = oscml.hpo.resume.start()
            logging.info('result=%s', result)

    
    """
    def test_rf_hpo_fixed_trial(self):

        testargs = ['test', 
            '--fixedtrial', 'True',
            '--dataset', 'HOPV15',
        ]
        with unittest.mock.patch('sys.argv', testargs):
            best_value = oscml.hpo.optunawrapper.start_hpo(
                    init=oscml.hpo.start_rf_with_hpo.init, 
                    objective=oscml.hpo.start_rf_with_hpo.objective, 
                    metric='mse', 
                    direction='minimize',
                    fixed_trial_params=oscml.hpo.start_rf_with_hpo.fixed_trial()
                )

    def test_rf_hpo_with_some_trials(self):

        testargs = ['test', 
            '--trials', '10',
            '--dataset', 'HOPV15',
        ]
        with unittest.mock.patch('sys.argv', testargs):
            best_value = oscml.hpo.optunawrapper.start_hpo(
                    init=oscml.hpo.start_rf_with_hpo.init, 
                    objective=oscml.hpo.start_rf_with_hpo.objective, 
                    metric='mse', 
                    direction='minimize',
                    fixed_trial_params=oscml.hpo.start_rf_with_hpo.fixed_trial()
                )

    def test_rf_hpo_with_fixed_trial_and_negative_mean_score(self):

        testargs = ['test', 
                '--fixedtrial', 'True',
                '--dataset', 'HOPV15',
            ]
        with unittest.mock.patch('sys.argv', testargs):
            best_value = oscml.hpo.optunawrapper.start_hpo(
                    init=oscml.hpo.start_rf_with_hpo.init, 
                    objective=oscml.hpo.start_rf_with_hpo.objective, 
                    metric='mse', 
                    direction='minimize',
                    fixed_trial_params={
                        'type': 'morgan', 'nBits': 256, 'radius': 5, 'useChirality': True, 'useBondTypes': True,
                        'n_estimators': 98, 'max_depth': 48, 'min_samples_split': 3, 'min_samples_leaf': 4, 'max_features': 1.0, 'bootstrap': False, 'max_samples': 10}

                )
    """

if __name__ == '__main__':

    #unittest.main()

    suite = unittest.TestSuite()
    #suite.addTest(Test_HPO('test_train_gnn_cep25000_one_trial'))
    #suite.addTest(Test_HPO('test_train_gnn_hopv15_one_trial'))
    #suite.addTest(Test_HPO('test_train_bilstm_cep25000_one_trial'))
    #suite.addTest(Test_HPO('test_train_bilstm_hopv15_one_trial'))
    #suite.addTest(Test_HPO('test_train_attentiveFP_cep25000_simple_featurizer'))
    #suite.addTest(Test_HPO('test_hpo_attentiveFP_hopv15_full_featurizer'))
    suite.addTest(Test_HPO('test_hpo_attentiveFP_hopv15_full_featurizer_with_config_file'))
    #suite.addTest(Test_HPO('test_load_model_from_checkpoint'))
    #suite.addTest(Test_HPO('test_gnn_cep25000_ckpt_test_only'))
    #suite.addTest(Test_HPO('test_gnn_cep25000_ckpt_resume_training'))
    #suite.addTest(Test_HPO('test_infinite_trials_and_time_out_gnn'))
    #suite.addTest(Test_HPO('test_infinite_trials_and_time_out_bilstm'))
    #suite.addTest(Test_HPO('test_rf_hpo_fixed_trial'))
    #suite.addTest(Test_HPO('test_rf_hpo_with_some_trials'))
    #suite.addTest(Test_HPO('test_rf_hpo_with_fixed_trial_and_negative_mean_score'))
    runner = unittest.TextTestRunner()
    runner.run(suite)