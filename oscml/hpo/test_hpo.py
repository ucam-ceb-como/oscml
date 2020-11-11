import logging
import os
import unittest
from unittest.mock import patch

import pytorch_lightning as pl

import oscml.hpo.optunawrapper
import oscml.hpo.start_bilstm_with_hpo
import oscml.hpo.start_gnn_with_hpo
import oscml.hpo.start_mnist_with_hpo
import oscml.utils.util

class Test_HPO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        oscml.utils.util.init_logging('.', './tmp')

    def test_train_mnist_with_fixed_trial(self):
        testargs = ['test', '--fixedtrial', 'True']
        with unittest.mock.patch('sys.argv', testargs):
            best_value = oscml.hpo.start_mnist_with_hpo.start()
            self.assertAlmostEqual(0.8828125, best_value, 4)

    def test_train_gnn_hopv15_with_fixed_trial(self):
        testargs = ['test', 
            '--fixedtrial', 'True',
            '--dataset', 'HOPV15',
            '--epochs', '2'
        ]
        with unittest.mock.patch('sys.argv', testargs):
            best_value = oscml.hpo.start_gnn_with_hpo.start()

    def test_train_bilstm_cepdb_with_fixed_trial(self):
        testargs = ['test', 
            '--fixedtrial', 'True',
            '--dataset', 'CEP25000',
            '--epochs', '1'
            ]
        with unittest.mock.patch('sys.argv', testargs):
            best_value = oscml.hpo.start_bilstm_with_hpo.start()

    def test_train_bilstm_hopv15_with_fixed_trial(self):
        testargs = ['test', 
            '--fixedtrial', 'True',
            '--dataset', 'HOPV15',
            '--epochs', '2'
            ]
        with unittest.mock.patch('sys.argv', testargs):
            best_value = oscml.hpo.start_bilstm_with_hpo.start()

    def test_load_model_from_checkpoint(self):

        #load model
        path = './res/test_checkpoint_gnn_cep25000_e10/last.ckpt'
        model_class = oscml.models.model_gnn.GNNSimple
        model = model_class.load_from_checkpoint(path)

        logging.info(model)
        self.assertEqual('Adam', model.optimizer)
        self.assertEqual(1, len(model.conv_modules))
        # 4 hidden layer with 4 ReLUs and 1 output layer
        self.assertEqual(10, len(model.mlp))
        self.assertEqual(20, model.mlp[3].in_features)
        self.assertEqual(10, model.mlp[3].out_features)

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
        self.assertAlmostEqual(2.727805930844451, result[0]['mse'], 4)

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
            self.assertAlmostEqual(2.830736983235294, result['mse'], 4)
    
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

if __name__ == '__main__':

    #unittest.main()

    
    suite = unittest.TestSuite()
    #suite.addTest(Test_HPO('test_train_mnist_with_fixed_trial'))
    #suite.addTest(Test_HPO('test_train_gnn_hopv15_with_fixed_trial'))
    suite.addTest(Test_HPO('test_train_bilstm_cepdb_with_fixed_trial'))
    #suite.addTest(Test_HPO('test_train_bilstm_hopv15_with_fixed_trial'))
    #suite.addTest(Test_HPO('test_load_model_from_checkpoint'))
    #suite.addTest(Test_HPO('test_gnn_cep25000_ckpt_test_only'))
    #suite.addTest(Test_HPO('test_gnn_cep25000_ckpt_resume_training'))
    runner = unittest.TextTestRunner()
    runner.run(suite)
    