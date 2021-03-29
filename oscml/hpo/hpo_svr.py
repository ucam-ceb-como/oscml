import logging
import oscml.models.model_kernel
from oscml.utils.util_config import set_config_param
from oscml.models.model_kernel import preprocess_data_phys_and_struct
from oscml.hpo.hpo_utils import preproc_training_params, BL_model_train
from oscml.hpo.hpo_utils import BL_model_train_cross_validate, BL_bestTrialRetrainDataPreproc
from oscml.hpo.objclass import Objective
from oscml.utils.util_sklearn import train_model, train_model_hpo, best_model_retraining

def getObjectiveSVR(modelName, data, config, logFile, logDir,
                    crossValidation, bestTrialRetraining=False, transferLearning=False):

    if bestTrialRetraining:
        data = BL_bestTrialRetrainDataPreproc(data)

    objectiveSVR = Objective(modelName=modelName, data=data, config=config,
                        logFile=logFile, logDir=logDir)

    model_trainer_func = BL_model_train_cross_validate if crossValidation else BL_model_train

    objectiveSVR.addPreModelCreateTask(objParamsKey='training', funcHandle=preproc_training_params)
    objectiveSVR.setModelCreator(funcHandle=model_create)

    if bestTrialRetraining:
        objectiveSVR.setModelTrainer(funcHandle=model_trainer_func, extArgs=[data_preproc, best_model_retraining])
    else:
        objectiveSVR.setModelTrainer(funcHandle=model_trainer_func, extArgs=[data_preproc, train_model_hpo])

    return objectiveSVR

def model_create(trial, data, objConfig, objParams):
    # set model parameters from the config file
    #--------------------------------------
    """
        'kernel'
        'gamma_structural'
    """
    model_conf = objConfig['config']['model']['model_specific']
    model_params = {}
    for key, value in model_conf.items():
        model_params.update({key: set_config_param(trial=trial,param_name=key,param=value, all_params=model_params)})

    logging.info('model params=%s', model_params)

    model = oscml.models.model_kernel.SVRWrapper(**model_params)

    return model

def data_preproc(trial, data, objConfig, objParams):
    # set fingerprint parameters from the config file
    #--------------------------------------
    fp_conf = objConfig['config']['model']['fingerprint_specific']

    fp_params = {}
    for key, value in objConfig['config']['model']['fingerprint_specific'].items():
        fp_params.update({key: set_config_param(trial=trial,param_name=key,param=value, all_params=fp_params)})

    logging.info('fingerprting params=%s', fp_params)

    x_column = objConfig['config']['dataset']['x_column'][0]
    y_column = objConfig['config']['dataset']['y_column'][0]

    data_processed = {
        'train': None,
        'val': None,
        'test': None,
        'scaler': None,
        'transformer': data['transformer']
    }

    x, y, scaler_svr_physical_data= preprocess_data_phys_and_struct(
            data['train'], fp_params, train_size=1, column_smiles=x_column,
            columns_phys=None, column_y=y_column)
    data_processed['train'] = (x, y)
    data_processed['scaler'] = scaler_svr_physical_data

    if data['val'] is not None:
        x, y, _= preprocess_data_phys_and_struct(
            data['val'], fp_params, train_size=1, column_smiles=x_column,
            columns_phys=None, column_y=y_column, scaler_svr_physical_data=scaler_svr_physical_data)
        data_processed['val'] = (x,y)

    x, y, _= preprocess_data_phys_and_struct(
        data['test'], fp_params, train_size=1, column_smiles=x_column,
        columns_phys=None, column_y=y_column, scaler_svr_physical_data=scaler_svr_physical_data)
    data_processed['test'] = (x,y)

    return data_processed