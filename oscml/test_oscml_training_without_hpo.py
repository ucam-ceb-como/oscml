
import logging
import os
import unittest

import pytorch_lightning as pl

import oscml.start
import oscml.start_bilstm
import oscml.start_gnn
import oscml.start_mnist
import oscml.test_oscml
import oscml.data.dataset
import oscml.data.dataset_cep
import oscml.data.dataset_hopv15
import oscml.models.model_bilstm
import oscml.models.model_gnn
import oscml.models.model_example_mlp_mnist
import oscml.utils.util
import oscml.utils.util_lightning

class Test_Oscml_Training_Without_HPO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        #oscml.utils.util.init_standard_logging()
        pass


    def assert_same_test_metrics_for_reloaded_model(self, model_class, model_instance, trainer, test_dl):
    
        metrics = trainer.test(model_instance, test_dataloaders=test_dl)[0]
        logging.info('expected metrics on test set=' + str(metrics))

        log_dir = trainer.logger.log_dir
        checkpoint_path = log_dir + '/checkpoints/last.ckpt'
        hparams_file = log_dir + '/hparams.yaml'
        model_reloaded = model_class.load_from_checkpoint(checkpoint_path=checkpoint_path, 
                hparams_file=hparams_file)
        metrics_reloaded = trainer.test(model_reloaded, test_dataloaders=test_dl)[0]
        logging.info('actual metrics on test set=' + str(metrics_reloaded))

        oscml.test_oscml.assertNearlyEqual(metrics['mse'], metrics_reloaded['mse'])
        oscml.test_oscml.assertNearlyEqual(metrics['r'], metrics_reloaded['r'])

    def test_train_mlp_mnist_without_hpo(self):
        oscml.start_mnist.start('.', '.', epochs=2)

    def test_train_gnn_hopv_without_hpo(self):
        model, model_instance, trainer, test_dl = oscml.start_gnn.start('.', '.', epochs=2)
        self.assert_same_test_metrics_for_reloaded_model(model, model_instance, trainer, test_dl)

    def test_train_bilstm_cepdb_without_hpo(self):
        model, model_instance, trainer, test_dl = oscml.start_bilstm.start('.', '.', epochs=2)
        self.assert_same_test_metrics_for_reloaded_model(model, model_instance, trainer, test_dl)

if __name__ == '__main__':

    #unittest.main()

    test = Test_Oscml_Training_Without_HPO()
    test.test_train_mlp_mnist_without_hpo()
    test.test_train_gnn_hopv_without_hpo()
    test.test_train_bilstm_cepdb_without_hpo()