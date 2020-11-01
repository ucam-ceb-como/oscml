PATH_ROOT = '.'
PATH_CEPDB_VALID_SMILES = PATH_ROOT + '/data/processed/CPEDB_valid_SMILES.csv'
PATH_CEPDB_25000 = PATH_ROOT + '/data/processed/CEPDB_25000.csv'
PATH_HOPV_15 = PATH_ROOT + '/data/raw/HOPV_15_revised_2.data'
PATH_OSAKA = PATH_ROOT + '/data/raw/Nagasawa_RF_SI.txt'

def assertNearlyEqual(actual, expected, epsilon=0.001):
    assert abs(actual - expected) < epsilon