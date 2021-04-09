import logging
from oscml.utils.util_config import set_config_param
from oscml.models.model_bilstm import get_dataloaders
from oscml.hpo.objclass import Objective
from oscml.hpo.hpo_utils import preproc_training_params
from oscml.utils.util_transfer_learning import BiLstmForPceTransfer
from oscml.models.model_bilstm import BiLstmForPce
from oscml.hpo.hpo_utils import NN_model_train, NN_model_train_cross_validate
from oscml.hpo.hpo_utils import NN_addBestModelRetrainCallback
from oscml.hpo.hpo_utils import NN_logBestTrialRetraining, NN_transferLearningCallback
from oscml.hpo.hpo_utils import NN_logTransferLearning, NN_prepareTransferLearningModel
from oscml.hpo.hpo_utils import NN_valDataCheck, NN_loadModelFromCheckpoint
from oscml.data.dataset import get_dataset_info

def getObjectiveBilstm(modelName, data, config, logFile, logDir,
                       crossValidation, bestTrialRetraining=False,
                       transferLearning=False, evaluateModel=False):

    if not crossValidation:
        # for not cv job, make sure there is non empty validation set
        # as NN methods require it for training
        data = NN_valDataCheck(data, config, transferLearning)

    # create a model agnostic objective instance
    objectiveBilstm = Objective(modelName=modelName, data=data, config=config,
                        logFile=logFile, logDir=logDir)

    # add goal and model specific settings
    if bestTrialRetraining:
        objectiveBilstm = addBestTrialRetrainingSettings(objectiveBilstm)
    elif transferLearning:
        objectiveBilstm = addTransferLearningSettings(objectiveBilstm, crossValidation, config)
    else:
        objectiveBilstm = addHpoSettings(objectiveBilstm, crossValidation)
    return objectiveBilstm

def addBestTrialRetrainingSettings(objective):
    objective.setModelCreator(funcHandle=model_create,extArgs=[BiLstmForPce])
    objective.setModelTrainer(funcHandle=NN_model_train,extArgs=[data_preproc])
    objective.addPreModelCreateTask(objParamsKey='training', funcHandle=preproc_training_params)
    objective.addPostModelCreateTask(objParamsKey='callbackBestTrialRetraining', funcHandle=NN_addBestModelRetrainCallback)
    objective.addPostTrainingTask(objParamsKey='logBestTrialRetrain', funcHandle=NN_logBestTrialRetraining)
    return objective

def addTransferLearningSettings(objective, crossValidation, config):
    freeze_and_train = config['transfer_learning']['freeze_and_train']
    modelCreatorClass = BiLstmForPceTransfer if freeze_and_train else BiLstmForPce
    model_trainer_func = NN_model_train_cross_validate if crossValidation else NN_model_train

    # this flag disables model creation in the objclass _createModel step, instead the model is
    # created in the trainer as part of the cross validation loop
    objective.setCrossValidation(crossValidation)
    objective.setModelCreator(funcHandle=model_create,extArgs=[modelCreatorClass])
    objective.setModelTrainer(funcHandle=model_trainer_func,extArgs=[data_preproc])
    objective.addPreModelCreateTask(objParamsKey='training', funcHandle=preproc_training_params)
    objective.addPostModelCreateTask(objParamsKey='callbackTransferLearning', funcHandle=NN_transferLearningCallback)
    objective.addPostModelCreateTask(objParamsKey='transferLearningModel', funcHandle=NN_prepareTransferLearningModel)
    objective.addPostTrainingTask(objParamsKey='logTransferLearning', funcHandle=NN_logTransferLearning)
    return objective

def addHpoSettings(objective, crossValidation):
    model_trainer_func = NN_model_train_cross_validate if crossValidation else NN_model_train

    # this flag disables model creation in the objclass _createModel step, instead the model is
    # created in the trainer as part of the cross validation loop
    objective.setCrossValidation(crossValidation)
    objective.setModelCreator(funcHandle=model_create,extArgs=[BiLstmForPce])
    objective.setModelTrainer(funcHandle=model_trainer_func,extArgs=[data_preproc])
    objective.addPreModelCreateTask(objParamsKey='training', funcHandle=preproc_training_params)
    return objective


def model_create(trial, data, objConfig, objParams, modelCreatorClass):
    transformer = data['transformer']
    type_dict = objConfig['config']['model']['type_dict']
    info = get_dataset_info(type_dict)
    number_subgraphs = info.number_subgraphs()
    max_sequence_length = objConfig['config']['model']['max_sequence_length']

    batch_size = objParams['training']['batch_size']
    optimizer = objParams['training']['optimiser']

    # set model parameters from the config file
    model_params = {}
    for key, value in objConfig['config']['model']['model_specific'].items():
        model_params.update({key: set_config_param(trial=trial,param_name=key,param=value, all_params=model_params)})

    # add output dimension to the mlp_units
    model_params['mlp_units'] = model_params.get('mlp_units', []) + [1]

    # add extra params not defined in the config file
    extra_params =  {
        # additional non-hyperparameter values
        'lstm_hidden_dim': model_params['embedding_dim'],
        'padding_index': 0,
        'target_mean': transformer.target_mean,
        'target_std': transformer.target_std,
        'number_of_subgraphs': number_subgraphs
    }
    model_params.update(extra_params)
    logging.info('model params=%s', model_params)

    model_params.pop('mlp_layers',None) # this is not needed for the model creation
    model = modelCreatorClass(**model_params, optimizer=optimizer)

    return model

def data_preproc(trial, data, objConfig, objParams):
    type_dict = objConfig['config']['model']['type_dict']
    df_train = data['train']
    df_val = data['val']
    df_test = data['test']
    transformer = data['transformer']

    info = get_dataset_info(type_dict)
    number_subgraphs = info.number_subgraphs()
    max_sequence_length = objConfig['config']['model']['max_sequence_length']
    batch_size = objParams['training']['batch_size']

    # dataloaders
    train_dl, val_dl, test_dl = get_dataloaders(type_dict, df_train, df_val, df_test,
        transformer, batch_size=batch_size, max_sequence_length=max_sequence_length)

    processed_data = {
        "train": train_dl,
        "val": val_dl,
        "test": test_dl,
        "transformer": transformer
    }
    return processed_data