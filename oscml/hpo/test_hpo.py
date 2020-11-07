import logging
import os
import unittest
from unittest.mock import patch

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
        testargs = ['test', '--fixedtrial', 'True']
        with unittest.mock.patch('sys.argv', testargs):
            best_value = oscml.hpo.start_gnn_with_hpo.start()
            # validation result of epoch 1:
            # {'phase': 'val', 'val_loss': 1.1001275086974058, 'epoch': 0, 'time': '2020-11-07 17:06:25.367178', 'mse': 6.064039445015627, 'rmse': 2.4625270445247147, 'R2': -0.06626423304785267, 'r': 0.27407959893772177, 'mae': 2.152695351953835, 'count': 30}
            self.assertAlmostEqual(1.1001275086974058, best_value, 4)

    def test_train_bilstm_cepdb_with_fixed_trial(self):
        testargs = ['test', '--fixedtrial', 'True']
        with unittest.mock.patch('sys.argv', testargs):
            best_value = oscml.hpo.start_bilstm_with_hpo.start()
            # validation result of epoch 1:
            # {'phase': 'val', 'val_loss': 0.5200955136203296, 'epoch': 0, 'time': '2020-11-07 16:44:55.760944', 'mse': 3.009651182848746, 'rmse': 1.7348346269453887, 'R2': 0.48011735180977666, 'r': 0.6936366438032565, 'mae': 1.3419126654810967, 'count': 5000}
            self.assertAlmostEqual(0.5200955136203296, best_value, 4)        

if __name__ == '__main__':

    #unittest.main()

    suite = unittest.TestSuite()
    #suite.addTest(Test_HPO('test_train_mnist_with_fixed_trial'))
    #suite.addTest(Test_HPO('test_train_gnn_hopv15_with_fixed_trial'))
    suite.addTest(Test_HPO('test_train_bilstm_cepdb_with_fixed_trial'))
    runner = unittest.TextTestRunner()
    runner.run(suite)