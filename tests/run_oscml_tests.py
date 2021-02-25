import test_hpo
import test_models
import test_config_parser
import test_data
import unittest

if __name__ == '__main__':

    suite = unittest.TestSuite()
    suite.addTest(test_hpo.Test_HPO('test_train_gnn_cep25000_one_trial'))
    suite.addTest(test_hpo.Test_HPO('test_train_gnn_hopv15_one_trial'))
    suite.addTest(test_hpo.Test_HPO('test_train_bilstm_cep25000_one_trial'))
    suite.addTest(test_hpo.Test_HPO('test_train_bilstm_hopv15_one_trial'))
    suite.addTest(test_hpo.Test_HPO('test_train_attentiveFP_cep25000_simple_featurizer'))
    suite.addTest(test_hpo.Test_HPO('test_hpo_attentiveFP_hopv15_full_featurizer'))
    suite.addTest(test_hpo.Test_HPO('test_hpo_attentiveFP_hopv15_full_featurizer_with_config_file'))
    suite.addTest(test_hpo.Test_HPO('test_train_rf_cep25000_one_trial_with_config_file'))
    suite.addTest(test_hpo.Test_HPO('test_train_svr_hopv15_one_trial_with_config_file'))
    suite.addTest(test_config_parser.TestConfigparser('test_config_parser'))
    suite.addTest(test_models.TestModels('test_bilstm_dataloader_for_hopv15'))
    suite.addTest(test_models.TestModels('test_bilstm_dataloader_for_cep25000'))
    suite.addTest(test_models.TestModels('test_profile_gnn_dataloader_for_cep25000'))
    suite.addTest(test_data.TestData('test_dataset_info_for_cepdb_25000'))
    suite.addTest(test_data.TestData('test_dataset_transform_cep_25000'))
    runner = unittest.TextTestRunner()
    runner.run(suite)
