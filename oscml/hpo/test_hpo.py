import logging
import os
import time
import unittest
from unittest.mock import patch

import pytorch_lightning as pl

import oscml.hpo.optunawrapper
import oscml.hpo.train
import oscml.hpo.start_bilstm_with_hpo
import oscml.hpo.start_gnn_with_hpo
import oscml.hpo.start_mnist_with_hpo
import oscml.hpo.start_rf_with_hpo
import oscml.utils.util


def create_config_template_optimizer():
    return {
        "optimizer": {
            'opt_name': 'Adam',             # Adam, SGD, RMSProp
            'lr': 0.001,
            'weight_decay': 0,
            'momentum': 0,              # SGD and RMSProp only
            'nesterov': False,          # SGD only
        }
    }

def create_config_template_bilstm():
    d = {
        "model_name": "BILSTM",
        "model": {
            'embedding_dim': 128,
            'mlp_layers': 3,
            'mlp_units_0': 60,
            'mlp_units_1': 20,
            'mlp_units_2': 20,
            'mlp_dropout': 0.1
        }
    }
    d.update(create_config_template_optimizer())
    return d

def create_config_template_simplegnn():
    d = {
        "model_name": "SimpleGNN",
        "model": {
            'embedding_dim':30,
            'conv_layers': 2,
            'conv_dims_0': 30,
            'conv_dims_1': 20, 
            'mlp_layers': 3,
            'mlp_units_0': 20,
            'mlp_units_1': 10,
            'mlp_units_2': 5,
            'mlp_dropout': 0.2,
        }
    }
    d.update(create_config_template_optimizer())
    return d

def create_config_template_attentivefp():
    d = {
        "model_name": "AttentiveFP",
        "model": {
            'graph_feat_size': 200,
            'num_layers': 4,
            'num_timesteps': 2,
            'dropout': 0.,
        }
    }
    d.update(create_config_template_optimizer())
    return d


class Test_HPO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        oscml.utils.util.init_logging('.', './tmp')

    def test_train_gnn_cep25000_with_fixed_trial(self):

        config = create_config_template_simplegnn()

        testargs = ['test', 
            '--fixedtrial', 'True',
            '--dataset', 'CEP25000',
            '--epochs', '1'
        ]
        with unittest.mock.patch('sys.argv', testargs):
            oscml.hpo.train.start(config_dev=config)

    def test_train_gnn_hopv15_with_fixed_trial(self):

        config = create_config_template_simplegnn()

        testargs = ['test', 
            '--fixedtrial', 'True',
            '--dataset', 'HOPV15',
            '--epochs', '1'
        ]
        with unittest.mock.patch('sys.argv', testargs):
            oscml.hpo.train.start(config_dev=config)

    def test_train_bilstm_cep25000_with_fixed_trial(self):

        config = create_config_template_bilstm()

        testargs = ['test', 
            '--fixedtrial', 'True',
            '--dataset', 'CEP25000',
            '--epochs', '1',
        ]
        with unittest.mock.patch('sys.argv', testargs):
            oscml.hpo.train.start(config_dev=config)

    def test_train_bilstm_hopv15_with_fixed_trial(self):

        config = create_config_template_bilstm()

        testargs = ['test', 
            '--fixedtrial', 'True',
            '--dataset', 'HOPV15',
            '--epochs', '2'
        ]
        with unittest.mock.patch('sys.argv', testargs):
            oscml.hpo.train.start(config_dev=config)

    def test_train_attentiveFP_cep25000_full_featurizer_with_fixed_trial(self):

        config = create_config_template_attentivefp()

        testargs = ['test', 
            '--fixedtrial', 'True',
            '--dataset', 'CEP25000',
            '--epochs', '1',
            ]
        with unittest.mock.patch('sys.argv', testargs):
            oscml.hpo.train.start(config_dev=config)
    
    def test_train_attentiveFP_cep25000_simple_featurizer_with_fixed_trial(self):

        config = create_config_template_attentivefp()

        testargs = ['test', 
            '--fixedtrial', 'True',
            '--dataset', 'CEP25000',
            '--epochs', '100',
            '--featurizer', 'simple'
            ]
        with unittest.mock.patch('sys.argv', testargs):
            oscml.hpo.train.start(config_dev=config)

    def test_hpo_attentiveFP_hopv15_full_featurizer(self):

        config = create_config_template_attentivefp()

        testargs = ['test', 
            '--dataset', 'HOPV15',
            '--epochs', '1',
            '--trials', '3'
            ]
        with unittest.mock.patch('sys.argv', testargs):
            oscml.hpo.train.start(config_dev=config)

    """
    def test_load_model_from_checkpoint(self):

        #load model
        path = './res/test_checkpoint_gnn_cep25000_e10/last.ckpt'
        model_class = oscml.models.model_gnn.GNNSimple
        model = model_class.load_from_checkpoint(path)

        logging.info(model)

        print(model.optimizer)

        self.assertEqual('Adam', model.optimizer['name'])
        self.assertEqual(3, len(model.conv_modules))
        # 4 hidden layer with 4 ReLUs and Dropout and 1 output layer
        self.assertEqual(10, len(model.mlp))
        # check second layer
        self.assertEqual(16, model.mlp[3].in_features)
        self.assertEqual(11, model.mlp[3].out_features)

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
        # the value is the validation result copied from the corresponding log file / metric.csv file
        self.assertAlmostEqual(2.7822956321521035, result[0]['mse'], 4)

    def test_gnn_cep25000_ckpt_test_only(self):
        testargs = ['test', 
            '--epochs', '0',
            '--ckpt',  './res/test_checkpoint_gnn_cep25000_e10/last.ckpt',
            '--dataset', oscml.data.dataset_cep.CEP25000
        ]

        with unittest.mock.patch('sys.argv', testargs):

            result = oscml.hpo.optunawrapper.start_hpo(
                init=None,
                objective=None,
                metric='val_loss', 
                direction='minimize',
                resume=oscml.hpo.start_gnn_with_hpo.resume
            )
            self.assertAlmostEqual(2.8787141593397987, result['mse'], 4)
    
    def test_gnn_cep25000_ckpt_resume_training(self):
        testargs = ['test', 
            '--epochs', '1',
            '--ckpt',  './res/test_checkpoint_gnn_cep25000_e10/last.ckpt',
            '--dataset', oscml.data.dataset_cep.CEP25000
        ]

        with unittest.mock.patch('sys.argv', testargs):

            value = oscml.hpo.optunawrapper.start_hpo(
                init=None,
                objective=None,
                metric='val_loss', 
                direction='minimize',
                resume=oscml.hpo.start_gnn_with_hpo.resume
            )

    def test_infinite_trials_and_time_out_gnn(self):
        testargs = ['test', 
            '--dataset', 'HOPV15',
            '--epochs', '1',
            '--timeout', '60'
        ]
        with unittest.mock.patch('sys.argv', testargs):
            best_value = oscml.hpo.start_gnn_with_hpo.start()

    def test_infinite_trials_and_time_out_bilstm(self):
        testargs = ['test', 
            '--dataset', 'CEP25000',
            '--epochs', '1',
            '--timeout', '60'
        ]
        with unittest.mock.patch('sys.argv', testargs):
            best_value = oscml.hpo.start_gnn_with_hpo.start()

    def objective_raising_error(self, trial):
        #time.sleep(1)
        raise RuntimeError('some fancy error')

    def test_objective_raising_error(self):

        testargs = ['test', 
            '--dataset', 'CEP25000',
            '--epochs', '1',
            '--timeout', '20'
        ]

        caught_error = False
        with unittest.mock.patch('sys.argv', testargs):
            try:
                best_value = oscml.hpo.optunawrapper.start_hpo(
                        init=None, 
                        objective=self.objective_raising_error, 
                        metric='val_loss', 
                        direction='minimize'
                    )
            except (RuntimeError, ValueError):
                caught_error = True
        
        self.assertEqual(True, caught_error)
    
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
    #suite.addTest(Test_HPO('test_train_gnn_cep25000_with_fixed_trial'))
    #suite.addTest(Test_HPO('test_train_gnn_hopv15_with_fixed_trial'))
    #suite.addTest(Test_HPO('test_train_bilstm_cep25000_with_fixed_trial'))
    #suite.addTest(Test_HPO('test_train_bilstm_hopv15_with_fixed_trial'))
    #suite.addTest(Test_HPO('test_train_attentiveFP_cep25000_full_featurizer_with_fixed_trial'))
    suite.addTest(Test_HPO('test_train_attentiveFP_cep25000_simple_featurizer_with_fixed_trial'))
    #suite.addTest(Test_HPO('test_hpo_attentiveFP_hopv15_full_featurizer'))
    #suite.addTest(Test_HPO('test_load_model_from_checkpoint'))
    #suite.addTest(Test_HPO('test_gnn_cep25000_ckpt_test_only'))
    #suite.addTest(Test_HPO('test_gnn_cep25000_ckpt_resume_training'))
    #suite.addTest(Test_HPO('test_infinite_trials_and_time_out_gnn'))
    #suite.addTest(Test_HPO('test_infinite_trials_and_time_out_bilstm'))
    #suite.addTest(Test_HPO('test_objective_raising_error'))
    #suite.addTest(Test_HPO('test_rf_hpo_fixed_trial'))
    #suite.addTest(Test_HPO('test_rf_hpo_with_some_trials'))
    #suite.addTest(Test_HPO('test_rf_hpo_with_fixed_trial_and_negative_mean_score'))
    runner = unittest.TextTestRunner()
    runner.run(suite)