
import os
import unittest

import pytorch_lightning as pl

import oscml.start_bilstm
import oscml.start_gnn
import oscml.test_oscml
import oscml.data.dataset
import oscml.data.dataset_cep
import oscml.data.dataset_hopv15
import oscml.models.model_bilstm
import oscml.models.model_gnn
import oscml.models.model_example_mlp_mnist
import oscml.start
import oscml.utils.util
import oscml.utils.util_lightning

class Test_Oscml_Training_Without_HPO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        oscml.utils.util.init_standard_logging()

    def test_train_mlp_mnist_without_hpo(self):

        csv_logger = oscml.utils.util.init_standard_logging()
        trainer_params = oscml.utils.util_lightning.get_standard_params_for_trainer_short()
        trainer_params.update({
            'max_epochs': 2,
            'logger': csv_logger
        })

        data_loader_fct = oscml.models.model_example_mlp_mnist.get_mnist

        data_loader_params = {
            'mnist_dir': './tmp', 
            'batch_size': 128
        }

        model = oscml.models.model_example_mlp_mnist.MlpWithLightning
        
        model_params =  {
            # model parameters
            'number_classes': 10, 
            'layers': 3,
            'units': [100, 50, 20], 
            'dropouts': [0.2, 0.2, 0.2],
            # optimizer parameters
            'optimizer': 'Adam', 
            'optimizer_lr': 0.0015
        }

        params = {
            'data_loader_fct': data_loader_fct,
            'data_loader_params': data_loader_params,
            'model': model,
            'model_params': model_params,
            'trainer_params': trainer_params
        }

        oscml.utils.util_lightning.fit_model(**params)

    def test_train_gnn_hopv_without_hpo(self):
        oscml.start_gnn.start('.', '.', epochs=1)

    def test_train_and_test_bilstm_cepdb_without_hpo(self):
        oscml.start_bilstm.start('.', '.', epochs=1, plot=False)

if __name__ == '__main__':

    #unittest.main()

    test = Test_Oscml_Training_Without_HPO()
    #test.test_train_mlp_mnist_without_hpo()
    test.test_train_gnn_hopv_without_hpo()
    #test.test_train_and_test_bilstm_cepdb_without_hpo()