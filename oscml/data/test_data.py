import logging
import unittest

import numpy as np
import pandas as pd

import oscml.data.dataset
import oscml.data.dataset_cep
import oscml.utils.util
from oscml.utils.util import smiles2mol


def assert_PCE_values(df_100, df):
    for i in range(len(df_100)):
        df_100_pce = df_100['id'].iloc[i]
        pce = df['id'].iloc[i]
        assert df_100_pce == pce

class TestData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        oscml.utils.util.init_logging('.', './tmp')

    def setUp(self):
        self.path_CEPDB = oscml.data.dataset.path_cepdb_valid_smiles()
        self.path_CEPDB_25000 = oscml.data.dataset.path_cepdb_25000()

    def test_dataset_read_cep_25000(self):
        df_train, df_val, df_test = oscml.data.dataset.read_and_split(self.path_CEPDB_25000)
        assert len(df_train) == 15000
        assert len(df_val) == 5000
        assert len(df_test) == 5000

    def test_dataset_transform_cep_25000(self):
        df_train, _, _ = oscml.data.dataset.read_and_split(self.path_CEPDB_25000)
        transformer = oscml.data.dataset.create_transformer(df_train, column_target='pce', column_x='SMILES_str')
        self.assertAlmostEqual(4.120434375131375, transformer.target_mean, 3)
        self.assertAlmostEqual(2.405561853258728, transformer.target_std, 3)

    def internal_test_no_randomness_in_preprocessing_CEP(self, file_path, threshold, number_samples, train_ratio, val_ratio, test_ratio):
        seed_1 = 100
        logging.info('preprocessing for seed=' + str(seed_1))
        np.random.seed(seed_1)
        transformer, _, _, _, df_train_plus_val, _ = oscml.data.dataset_cep.preprocess_CEP(file_path, threshold, 
                                    number_samples, train_ratio, val_ratio, test_ratio)
        df_100 = df_train_plus_val.copy()
        df_100_pce_mean = transformer.target_mean
        df_100_pce_std = transformer.target_std
        # df_100 is just a copy of df_train_plus_val; thus their PCE values must be the same
        assert_PCE_values(df_100, df_train_plus_val)
        
        seed_2 = 200
        logging.info('preprocessing for seed=' + str(seed_2))
        np.random.seed(seed_2)
        transformer, _, _, _, df_train_plus_val, _ = oscml.data.dataset_cep.preprocess_CEP(file_path, threshold, 
                                    number_samples, train_ratio, val_ratio, test_ratio)    
        # very very unlikely to be the same; thus, the compare method will throw an error
        exception = None
        try:
            assert_PCE_values(df_100, df_train_plus_val)
        except AssertionError as exc:
            exception = exc
        assert exception
        
        np.random.seed(seed_1)
        logging.info('preprocessing for seed=' + str(seed_1))
        transformer, _, _, _, df_train_plus_val, _ = oscml.data.dataset_cep.preprocess_CEP(file_path, threshold, 
                                    number_samples, train_ratio, val_ratio, test_ratio)    
        # check that PCE values coincide with the result of the first run for the same seed
        assert_PCE_values(df_100, df_train_plus_val)
        assert df_100_pce_mean == transformer.target_mean
        assert df_100_pce_std == transformer.target_std  

    def test_no_randomness_in_preprocessing_CEP(self):
        """
        The following methods will call the method preprocess_CEP three times 
        with seed 100, 200, and again 100 and compare the first and third result. 
        An error occurs if both results don't coincide.
        """
        args = {
            'threshold': 0.0001,
            'number_samples': 25000,
            'train_ratio': 0.6,
            'val_ratio': 0.2,
            'test_ratio': 0.2}
        self.internal_test_no_randomness_in_preprocessing_CEP(
            self.path_CEPDB, **args)

    def test_dataset_update_state(self):

        mol2seq = oscml.features.weisfeilerlehman.Mol2seq_WL(radius=1)
        info = oscml.data.dataset.DatasetInfo(mol2seq)
        
        smiles = '[SiH2]1C=c2c3cc([se]c3c3cc4ccccc4cc3c2=C1)-c1cncs1'
        mol = smiles2mol(smiles)
        info.update(mol, smiles)
        assert info.max_molecule_size == 38
        assert info.max_smiles_length == 50
        assert len(info.mol2seq.fragment_dict) == 16
        assert len(info.node_types) == 7

        smiles = '[SiH2]1cc2cccc(-c3ccc(-c4scc5[nH]ccc45)c4nsnc34)c2c1'
        mol = smiles2mol(smiles)
        info.update(mol, smiles)
        assert info.max_molecule_size == 39
        assert info.max_smiles_length == 52
        assert len(info.node_types) == 7
        
    def test_dataset_skip_invalid_smiles(self):
        df = pd.read_csv(self.path_CEPDB)
        # all invalid SMIlES are in rows between row number 250000 and 300000
        # restrict to a small subset to speed up the test
        df = df[258000:259000]
        df_cleaned, info = oscml.data.dataset_cep.skip_invalid_smiles(
                df, 'SMILES_str')
        # number of invalid SMILES 
        assert 1000 - len(df_cleaned) == 139
        assert info.max_molecule_size == 51
        assert info.max_smiles_length == 63
        assert len(info.mol2seq.fragment_dict) == 54

    def test_sample_without_replacement(self):
        df = pd.read_csv(self.path_CEPDB)
        df_cleaned = oscml.data.dataset_cep.skip_all_small_pce_values(df.copy(), 0.0001)
        df_train, _ = oscml.data.dataset_cep.sample_without_replacement(df_cleaned, number_samples=1000, step=1.)
        assert len(df_train) == 1000

        df_cleaned = oscml.data.dataset_cep.skip_all_small_pce_values(df.copy(), 0.0001)
        df_train, df_val, df_test = oscml.data.dataset_cep.sample_without_replacement(df_cleaned, number_samples=[1000, 200, 300], step=.2)
        assert len(df_train) == 1000
        assert len(df_val) == 200
        assert len(df_test) == 300

    def store_CEP_cleaned_and_stratified(self):
        df = oscml.data.dataset_cep.store_CEP_cleaned_and_stratified(
            self.path_CEPDB, dst=None, number_samples=[15000, 5000, 5000], threshold_skip=0.0001)
        assert len(df) == 25000
        mask = (df['ml_phase'] == 'train')
        assert len(df[mask]) ==15000


if __name__ == '__main__':
    #unittest.main()
  
    suite = unittest.TestSuite()
    suite.addTest(TestData('store_CEP_cleaned_and_stratified'))
    runner = unittest.TextTestRunner()
    runner.run(suite)