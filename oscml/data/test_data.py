import unittest

import pandas as pd
from tqdm import tqdm

import oscml.data.dataset
import oscml.data.dataset_cep
import oscml.data.dataset_hopv15
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
        self.assertAlmostEqual(4.120434375131375, transformer.target_mean, 1)
        self.assertAlmostEqual(2.405561853258728, transformer.target_std, 1)

    def test_dataset_update_state(self):

        mol2seq = oscml.features.weisfeilerlehman.Mol2seq_WL(radius=1)
        info = oscml.data.dataset.DatasetInfo(mol2seq=mol2seq)
        
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

    def test_dataset_info_for_cepdb_25000(self):

        # check the correct size of dictionaries
        info = oscml.data.dataset_cep.create_dataset_info_for_CEP25000()
        number_node_types = len(info.node_types)
        self.assertEqual(8, number_node_types)
        number_fragment_types = len(info.mol2seq.fragment_dict)
        self.assertEqual(56, number_fragment_types)

        # read subset from CEPDB
        df = pd.read_csv(self.path_CEPDB_25000)
        for i in tqdm(range(len(df))):
            smiles = df.iloc[i]['SMILES_str']
            m = smiles2mol(smiles)
            info.update(m, smiles)

        # check that there are no additional node or fragment types
        number_node_types = len(info.node_types)
        self.assertEqual(8, number_node_types)
        number_fragment_types = len(info.mol2seq.fragment_dict)
        self.assertEqual(56, number_fragment_types)

    def test_dataset_info_for_hopv15(self):
        # check the correct size of dictionaries
        info = oscml.data.dataset_hopv15.create_dataset_info_for_HOPV15()
        number_node_types = len(info.node_types)
        self.assertEqual(12, number_node_types)
        number_fragment_types = len(info.mol2seq.fragment_dict)
        self.assertEqual(150, number_fragment_types)       
        
        # the fragments and node type were added to existing ones from CEP DB
        # compare the results when starting from scratich
        path = oscml.data.dataset.path_hopv_15()
        info_from_scratch = oscml.data.dataset_hopv15.generate_dictionaries(path, 'smiles', None)
        number_fragment_types = len(info_from_scratch.mol2seq.fragment_dict)
        self.assertEqual(134, number_fragment_types)       
        # that means there are 16 fragments in CEP DB that are not used in HOPV15

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

    def test_store_CEP_cleaned_and_stratified(self):
        df = oscml.data.dataset_cep.store_CEP_cleaned_and_stratified(
            self.path_CEPDB, dst=None, number_samples=[15000, 5000, 5000], threshold_skip=0.0001)
        assert len(df) == 25000
        mask = (df['ml_phase'] == 'train')
        assert len(df[mask]) ==15000

    def test_add_k_fold_columns(self):
        file = './data/processed/HOPV_15_revised_2_processed_homo.csv'
        df = pd.read_csv(file)
        k = 5
        oscml.data.dataset.add_k_fold_columns(df, k, seed=200, column_name_prefix='ml_phase')
        size = len(df)
        mask = [False]*size
        for i in range(k):
            column = 'ml_phase_fold_' + str(i)
            mask = (mask | (df[column] == 'test'))
        assert all(mask)

if __name__ == '__main__':
    unittest.main()
  
    #suite = unittest.TestSuite()
    #suite.addTest(TestData('test_dataset_info_for_cepdb_25000'))
    #suite.addTest(TestData('test_dataset_info_for_hopv15'))
    #suite.addTest(TestData('test_dataset_transform_cep_25000'))
    #suite.addTest(TestData('test_dataset_skip_invalid_smiles'))
    #suite.addTest(TestData('test_add_k_fold_columns'))
    #runner = unittest.TextTestRunner()
    #runner.run(suite)