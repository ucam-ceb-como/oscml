import logging
import unittest

import numpy as np
import pandas as pd

import oscml.data.dataset
import oscml.data.dataset_cep
import oscml.data.dataset_hopv15
import oscml.features.fingerprint
import oscml.features.fingerprint_ertl_ifg
import oscml.utils.util
from oscml.utils.util import smiles2mol

class TestFeatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        oscml.utils.util.init_logging('.', './tmp')

    def test_fingerprints(self):

        dataset_config = {
            "src": "./data/processed/CEP25000.csv",
            "z-stand": "False",
            "x_column": ["smiles"],
            "y_column": ["pce"],
            "type_dict": oscml.data.dataset_hopv15.HOPV15
        }
        df, _, _, _ = oscml.data.dataset.get_dataframes(dataset=dataset_config, src='.', train_size=100, test_size=30)

        df['rdkitmol'] = oscml.utils.util.smiles2mol_df(df, 'smiles')
        fg_ertl_dictionary = oscml.features.fingerprint.create_functional_group_ertl_dictionary(df)
        
        params = [
            # by default morgan
            {'radius':2, 'nBits':2048, 'useChirality':False, 'useBondTypes':True},
            {'type':'morgan', 'nBits':2048, 'radius':2, 'useChirality':False, 'useBondTypes':True},
            {'type':'topological_torsion', 'nBits':2048}, 
            {'type':'atom_pair', 'nBits':2048},
            {'type':'avalon', 'nBits':2048},
            {'type':'rdkit', 'length':2048},
            {'type':'mhfp', 'length':2048, 'radius':3}, 
            {'type': 'ertl', 'functional_group_dictionary': fg_ertl_dictionary}
        ]

        for p in params:
            oscml.features.fingerprint.get_fingerprints(df, 'rdkitmol', p, as_numpy_array = False)
            oscml.features.fingerprint.get_fingerprints(df, 'rdkitmol', p, as_numpy_array = True)


if __name__ == '__main__':
    unittest.main()
  
    #suite = unittest.TestSuite()
    #suite.addTest(TestData('test_fingerprints'))
    #runner = unittest.TextTestRunner()
    #runner.run(suite)