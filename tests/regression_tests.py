import logging
import os
import time
import unittest
from unittest.mock import patch

import pytorch_lightning as pl

import oscml.data.dataset_cep
import oscml.data.dataset_hopv15
import oscml.hpo.optunawrapper

from oscml.jobhandling import JobHandler
import oscml.utils.util
import glob
import pandas as pd
import contextlib
import shutil

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

test_cases = {
    #'rf': ['trials_5_no_cv','trials_1_cv_2', 'trials_2_cv_2'],
    #'svr': ['trials_5_no_cv','trials_1_cv_2', 'trials_2_cv_2'],
    #'simplegnn': ['trials_5_no_cv','trials_1_cv_2', 'trials_2_cv_2'],
    'bilstm': ['trials_5_no_cv','trials_1_cv_2', 'trials_2_cv_2']
}


rf_test_cases = {
    'rf': ['trials_5_no_cv','trials_1_cv_2', 'trials_2_cv_2']
    }
svr_test_cases = {
    'svr': ['trials_5_no_cv','trials_1_cv_2', 'trials_2_cv_2']
    }
simplegnn_test_cases = {
    'simplegnn': ['trials_5_no_cv','trials_1_cv_2', 'trials_2_cv_2']
    }
bilstm_test_cases = {
    'bilstm': ['trials_5_no_cv','trials_1_cv_2', 'trials_2_cv_2']
    }
attentivefp_test_cases = {
    'attentivefp': ['trials_5_no_cv','trials_1_cv_2', 'trials_2_cv_2']
    }


class Test_HPO(unittest.TestCase):
    def _run_test(self, test_cases):
        for model, subtests in test_cases.items():
            for test in subtests:
                print('========================================================')
                print('MODEL: ', model)
                print('TEST: ', test)
                print()
                print()
                test_path = os.path.join(THIS_DIR, 'modelsRegressionTests', model, test)
                input_file = os.path.join(test_path,'input.json')
                clean_test_dir(test_path)

                jobHandler = JobHandler({'<configFile>': input_file})
                jobHandler.runJob()
                compareResults(test_path, model, test)
                print('========================================================')
                print()
                print()

    def rf_test_suite(self):
        self._run_test(rf_test_cases)

    def svr_test_suite(self):
        self._run_test(svr_test_cases)

    def simplegnn_test_suite(self):
        self._run_test(simplegnn_test_cases)

    def bilstm_test_suite(self):
        self._run_test(bilstm_test_cases)

    def attentivefp_test_suite(self):
        self._run_test(attentivefp_test_cases)

def clean_test_dir(testDir):
    reg_test_dir = os.path.join(testDir,'reg_test')
    reg_test_db = os.path.join(testDir,'reg_test.db')
    if os.path.exists(reg_test_dir):
        shutil.rmtree(reg_test_dir)

    with contextlib.suppress(FileNotFoundError):
        os.remove(reg_test_db)

def compareResults(testDir, model, test):
    msg_head = "model: "+model+" test: "+test+" "
    # prepare ref data
    ref_hpo_file = os.path.join(testDir,'hpo_result.csv')
    ref_best_trial_file = glob.glob(os.path.join(testDir,'best_trial_retrain_model*.csv'))[0]
    ref_best_trial_nr = ref_best_trial_file.split('model_trial_')[1].split('.')[0]

    ref_hpo_df = pd.read_csv(ref_hpo_file)
    ref_hpo_df = ref_hpo_df.drop(['datetime_start', 'datetime_complete','duration'], axis=1)
    ref_best_trial_df = pd.read_csv(ref_best_trial_file)

    # prepare reg data
    reg_hpo_file = os.path.join(testDir,'reg_test','hpo','hpo_result.csv')
    reg_best_trial_nr = glob.glob(os.path.join(testDir,'reg_test','best_trial_retrain','trial_'+ref_best_trial_nr))
    assert len(reg_best_trial_nr) == 1, msg_head+"best trials numbers do not match!"

    reg_best_trial_file = os.path.join(reg_best_trial_nr[0],'best_trial_retrain_model.csv')

    reg_hpo_df = pd.read_csv(reg_hpo_file)
    reg_hpo_df = reg_hpo_df.drop(['datetime_start', 'datetime_complete','duration'], axis=1)
    reg_best_trial_df = pd.read_csv(reg_best_trial_file)

    assert ref_hpo_df.equals(reg_hpo_df)==True, msg_head+"test and reference hpo results are not equal"
    assert reg_best_trial_df.equals(reg_best_trial_df)==True, msg_head+"test and reference best trial retrain results are not equal"
    pass


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(Test_HPO('rf_test_suite'))
    suite.addTest(Test_HPO('svr_test_suite'))
    suite.addTest(Test_HPO('simplegnn_test_suite'))
    suite.addTest(Test_HPO('bilstm_test_suite'))
    suite.addTest(Test_HPO('attentivefp_test_suite'))
    runner = unittest.TextTestRunner()
    runner.run(suite)