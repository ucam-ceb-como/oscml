
import logging
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
        #oscml.utils.util.init_standard_logging()
        pass

    def test_train_mlp_mnist_without_hpo(self):

         # initialize, e.g. logging
        csv_logger = oscml.utils.util.init_logging(src_directory='.', dst_directory='.')
        logging.info('current working directory=' + os.getcwd())


        # define data loaders and params
        data_loader_fct = oscml.models.model_example_mlp_mnist.get_mnist

        data_loader_params = {
            'mnist_dir': './tmp', 
            'batch_size': 128
        }


        # define models and params
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


        # define params for Lightning trainer
        trainer_params = oscml.utils.util_lightning.get_standard_params_for_trainer_short()
        trainer_params.update({
            'max_epochs': 2,
            'logger': csv_logger
        })

        
        # create the model, dataloaders and Lightning trainer
        model_instance = model(**model_params)
        train_dl, val_dl = data_loader_fct(**data_loader_params)
        trainer = pl.Trainer(**trainer_params)


        # train the model
        trainer.fit(model_instance, train_dataloader=train_dl, val_dataloaders=val_dl)


        # test
        logging.info('start testing')
        # the mnist example didn't provide a test test. Thus, we will use
        # the validation dataloader as test loader here.
        metrics = trainer.test(model_instance, test_dataloaders=val_dl)
        logging.info(metrics)
    



    def test_train_gnn_hopv_without_hpo(self):
        oscml.start_gnn.start('.', '.', epochs=1)

    def test_train_and_test_bilstm_cepdb_without_hpo(self):
        oscml.start_bilstm.start('.', '.', epochs=1, plot=False)

if __name__ == '__main__':

    #unittest.main()

    test = Test_Oscml_Training_Without_HPO()
    test.test_train_mlp_mnist_without_hpo()
    #test.test_train_gnn_hopv_without_hpo()
    #test.test_train_and_test_bilstm_cepdb_without_hpo()