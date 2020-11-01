import unittest

import torch

import oscml.test_oscml  
import oscml.data.dataset
import oscml.data.dataset_cep
import oscml.data.dataset_hopv15
import oscml.models.model_bilstm
import oscml.utils.util
from oscml.utils.util import log, smiles2mol

class TestModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        oscml.utils.util.init_standard_logging()

    def test_out_of_vocabulary(self):
        """
        This test starts with the fragments of CEPDB (which contains 8 different atom types)
        and allows updating fragments on-the-fly by setting oov=True.
        Run the fragment mapping on HOPV15 and assert that 340 molecules from HOPV15 
        contain different fragment types and the number of different atom types
        has increased to 12.
        """
        df_hopv15 = oscml.data.dataset_hopv15.read(oscml.test_oscml.PATH_HOPV_15)
        mol2seq_without_OOV = oscml.data.dataset_cep.mol2seq_precalculated_with_OOV(None, radius=1, oov=False, column_smiles='smiles')
        mol2seq_with_OOV = oscml.data.dataset_cep.mol2seq_precalculated_with_OOV(None, radius=1, oov=True, column_smiles='smiles')
        log(mol2seq_with_OOV.atom_dict)
        assert len(mol2seq_with_OOV.atom_dict) == 8
        
        max_index = len(oscml.data.dataset_cep.WL_R1_FRAGMENT_DICT)-1
        count_larger = 0
        for i in range(len(df_hopv15)):
            smiles = df_hopv15.iloc[i]['smiles']
            m = smiles2mol(smiles)
            #print(smiles, m)
            seq_without_OOV = mol2seq_without_OOV(m)
            seq_with_OOV = mol2seq_with_OOV(m)
            #print(seq_without_OOV)
            #print('MY SEQ WITH OOV', seq_with_OOV)
            assert len(seq_without_OOV) == len(seq_with_OOV)
            larger_max_index = False
            for i in range(len(seq_without_OOV)):
                fragment_index = seq_without_OOV[i]
                if (fragment_index > max_index):
                    larger_max_index = True
                    # check that the index of the fragment was set to -1 (for OOV)
                    assert seq_with_OOV[i] == -1
                else:
                    assert seq_with_OOV[i] == fragment_index
            if larger_max_index:
                count_larger += 1
        
        log(mol2seq_with_OOV.atom_dict)
        assert len(mol2seq_with_OOV.atom_dict) == 12
        
        log('number of molecules with at least one new atom type=', count_larger)
        assert count_larger == 340

    def test_bilstm_model_forward(self
    ):
        df_train, df_val, df_test = oscml.data.dataset.read_and_split(oscml.test_oscml.PATH_CEPDB_25000)
        transformer = oscml.data.dataset.create_transformer(df_train, column_target='pce', column_x='SMILES_str')
        mol2seq = oscml.data.dataset_cep.mol2seq_precalculated_with_OOV(None, radius=1, oov=True)

        model_params =  {
            'number_of_subgraphs': 60,
            'subgraph_embedding_dim': 128,
            'mlp_dim_list': [256, 32, 32, 32, 1],
            'padding_index': 0,
            'target_mean': transformer.target_mean, 
            'target_std': transformer.target_std,
            'learning_rate': 0.001,
        }

        model = oscml.models.model_bilstm.BiLstmForPce(**model_params)

        batch = torch.LongTensor([[1,2,3], [4,5,6]]) #.to(device)
        output = model(batch)
        print(output)

if __name__ == '__main__':
    unittest.main()