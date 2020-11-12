import collections
import logging
from time import sleep

import numpy as np
import pandas as pd
import rdkit
import rdkit.Avalon.pyAvalonTools
import rdkit.Chem
import rdkit.Chem.AllChem
import rdkit.Chem.rdMHFPFingerprint
import sklearn

import oscml.features.fingerprint_ertl_ifg


def create_functional_group_ertl_dictionary(df):

    # key = fragment type
    # [id, number of molecules containing fragment type, overall number of fragment type]
    d = collections.defaultdict(lambda: [len(d), 0,0])

    for index, row in df.iterrows():
        mol = row['rdkitmol']
        fgs = oscml.features.fingerprint_ertl_ifg.identify_functional_groups(mol)
        #print(fgs)
        types = []
        for fg in fgs:
            if not fg.type in types:
                types.append(fg.type)
                d[fg.type][1] += 1
            d[fg.type][2] += 1

    return d

def get_fingerprint_ertl(mol, functional_group_dictionary):

    fingerprint = [0]*len(functional_group_dictionary)
    fgs = oscml.features.fingerprint_ertl_ifg.identify_functional_groups(mol)
    #print(fgs)
    for fg in fgs:
        identifier = functional_group_dictionary[fg.type][0]
        fingerprint[identifier] = 1

    return fingerprint

def get_fingerprint_MHFP(mol, length, radius):
    seed = 42
    encoder = rdkit.Chem.rdMHFPFingerprint.MHFPEncoder(length, seed)
    # returns an array of integers (no bits) of given length
    fp = encoder.EncodeMol(mol, radius=radius, rings=True, kekulize=True, min_radius=1)
    fp = np.array(fp)
    return fp

def create_fingerprint_function(params_fingerprint):

    params = params_fingerprint.copy()
    type = params.pop('type') if 'type' in params else 'morgan'
    if type == 'morgan':
        return lambda mol : rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect(mol, **params)
    elif type == 'atom_pair':
        return lambda mol : rdkit.Chem.rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, **params)
    elif type == 'topological_torsion':
        return lambda mol : rdkit.Chem.rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, **params)
    elif type == 'avalon':
        return lambda mol : rdkit.Avalon.pyAvalonTools.GetAvalonFP(mol, **params)
    elif type == 'rdkit':
        if 'length' in params:
            params['fpSize'] = params.pop('length')
        return lambda mol : rdkit.Chem.rdmolops.RDKFingerprint(mol, **params)
    elif type == 'mhfp':
        return lambda mol : get_fingerprint_MHFP(mol, **params)
    elif type =='ertl':
        return lambda mol : get_fingerprint_ertl(mol, **params)
    else:
        raise RuntimeError('unknown fingerprint type=' + type)

def get_fingerprints(df, column, params_fingerprint, as_numpy_array = True):
    logging.info('generating fingerprints, params=' + str(params_fingerprint))
    fp_func = create_fingerprint_function(params_fingerprint)
    x = []
    for i in range(len(df)):
        m = df.iloc[i][column]
        fingerprint = fp_func(m)
        if (as_numpy_array):
            fingerprint = np.array(fingerprint, dtype=np.float32)
        x.append(fingerprint)
    return x

def normalize_fingerprints(df, column):
    st = sklearn.preprocessing.StandardScaler()
    #x = np.array(list(df[column]))
    x = list(df[column])
    x = st.fit_transform(x)
    return list(x)