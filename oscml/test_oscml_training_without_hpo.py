
import os
import unittest

import pytorch_lightning as pl

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
            'mnist_dir': oscml.test_oscml.PATH_ROOT + '/tmp', 
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

        csv_logger = oscml.utils.util.init_standard_logging()
        trainer_params = oscml.utils.util_lightning.get_standard_params_for_trainer_short()
        trainer_params.update({
            'max_epochs': 2,
            'logger': csv_logger
        })

        df = oscml.data.dataset_hopv15.read(oscml.test_oscml.PATH_HOPV_15)
        df = oscml.data.dataset.clean_data(df, None, 'smiles', 'pce')

        df_train, df_val, df_test, transformer = oscml.data.dataset.split_data_frames_and_transform(
                df, column_smiles='smiles', column_target='pce', train_size=283, test_size=30)

        node2index = oscml.data.dataset_hopv15.ATOM_TYPES_HOPV15
        mol2seq = oscml.models.model_gnn.Mol2seq_simple(node2index, fix=True, oov=True)

        data_loader_fct = oscml.models.model_gnn.get_dataloaders

        data_loader_params = {
            'train': df_train,
            'val': df_val,
            'test': None, 
            'transformer': transformer,
            'batch_size': 20, 
            'mol2seq': mol2seq
        }

        model = oscml.models.model_gnn.GNNSimple

        model_params =  {
            'node_type_number': len(node2index),
            'conv_dim_list': [10, 10, 10],
            'mlp_dim_list': [10, 1],
            'padding_index': 0,
            'target_mean': transformer.target_mean, 
            'target_std': transformer.target_std,
            'learning_rate': 0.001,
        }

        params = {
            'data_loader_fct': data_loader_fct,
            'data_loader_params': data_loader_params,
            'model': model,
            'model_params': model_params,
            'trainer_params': trainer_params
        }

        oscml.utils.util_lightning.fit_model(**params)

    def test_train_bilstm_cepdb_without_hpo(self):

        csv_logger = oscml.utils.util.init_standard_logging()
        trainer_params = oscml.utils.util_lightning.get_standard_params_for_trainer_short()
        trainer_params.update({
            'max_epochs': 2,
            'logger': csv_logger
        })
       
        df_train, df_val, df_test = oscml.data.dataset.read_and_split(oscml.test_oscml.PATH_CEPDB_25000)
        transformer = oscml.data.dataset.create_transformer(df_train, column_target='pce', column_x='SMILES_str')
        mol2seq = oscml.data.dataset_cep.mol2seq_precalculated_with_OOV(None, radius=1, oov=True)

        data_loader_fct = oscml.models.model_bilstm.get_dataloaders

        data_loader_params = {
            'train': df_train,
            'val': df_val,
            'test': None, 
            'batch_size': 250, 
            'mol2seq': mol2seq,
            'max_sequence_length': 60,
            'padding_index': 0,
            'smiles_fct': transformer.transform_x,
            'target_fct': transformer.transform, 
        }

        model = oscml.models.model_bilstm.BiLstmForPce

        model_params =  {
            'number_of_subgraphs': 60,
            'subgraph_embedding_dim': 128,
            'mlp_dim_list': [256, 32, 32, 32, 1],
            'padding_index': 0,
            'target_mean': transformer.target_mean, 
            'target_std': transformer.target_std,
            'learning_rate': 0.001,
        }

        params = {
            'data_loader_fct': data_loader_fct,
            'data_loader_params': data_loader_params,
            'model': model,
            'model_params': model_params,
            'trainer_params': trainer_params
        }

        oscml.utils.util_lightning.fit_model(**params)


if __name__ == '__main__':

    unittest.main()

    #test = Test_Oscml_Training_Without_HPO()
    #test.test_train_mlp_mnist_without_hpo()
    #test.test_train_gnn_hopv_without_hpo()
    #test.test_train_bilstm_cepdb_without_hpo()