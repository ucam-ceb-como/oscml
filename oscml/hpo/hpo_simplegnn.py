import logging

import oscml.data.dataset
import oscml.models.model_gnn
from oscml.utils.util_config import set_config_param
import oscml.utils.util_transfer_learning
from oscml.hpo.hpo_utils import preproc_training_params
from oscml.utils.util_transfer_learning import SimpleGNNTransfer
from oscml.models.model_gnn import SimpleGNN
from oscml.hpo.hpo_utils import NN_model_train, NN_logBestTrialRetraining, NN_valDataCheck
from oscml.hpo.hpo_utils import NN_model_train_cross_validate, NN_addBestModelRetrainCallback
from oscml.hpo.hpo_utils import NN_prepareTransferLearningModel, NN_logTransferLearning, NN_transferLearningCallback
from oscml.hpo.objclass import Objective


def getObjectiveSimpleGNN(modelName, data, config, logFile, logDir,
                         crossValidation, bestTrialRetraining=False, transferLearning=False):

    if not crossValidation:
        # for not cv job, make sure there is non empty validation set
        # as NN methods require it for training
        data = NN_valDataCheck(data, config, transferLearning)

    # create SimpleGNN objective
    objectiveSimpleGNN = Objective(modelName=modelName, data=data, config=config,
                        logFile=logFile, logDir=logDir)

    # configure SimpleGNN objective
    if transferLearning and config['transfer_learning']['freeze_and_train']:
        modelCreatorClass = SimpleGNNTransfer
    else:
        modelCreatorClass = SimpleGNN

    model_trainer_func = NN_model_train_cross_validate if crossValidation else NN_model_train

    objectiveSimpleGNN.addPreModelCreateTask(objParamsKey='training', funcHandle=preproc_training_params)
    objectiveSimpleGNN.setModelCreator(funcHandle=model_create, extArgs=[modelCreatorClass])
    objectiveSimpleGNN.setModelTrainer(funcHandle=model_trainer_func, extArgs=[data_preproc])

    # this flag disables model creation in the objclass _createModel step, instead the model is
    # created in the trainer as part of the cross validation loop
    objectiveSimpleGNN.setCrossValidation(crossValidation)

    if bestTrialRetraining:
        objectiveSimpleGNN.addPostModelCreateTask(objParamsKey='callbackBestTrialRetraining', funcHandle=NN_addBestModelRetrainCallback)
        objectiveSimpleGNN.addPostTrainingTask(objParamsKey='logBestTrialRetrain', funcHandle=NN_logBestTrialRetraining)

    if transferLearning:
        objectiveSimpleGNN.addPostModelCreateTask(objParamsKey='callbackTransferLearning', funcHandle=NN_transferLearningCallback)
        objectiveSimpleGNN.addPostModelCreateTask(objParamsKey='transferLearningModel', funcHandle=NN_prepareTransferLearningModel)
        objectiveSimpleGNN.addPostTrainingTask(objParamsKey='logTransferLearning', funcHandle=NN_logTransferLearning)

    return objectiveSimpleGNN


def model_create(trial, data, objConfig, objParams, modelCreatorClass):
    transformer = data['transformer']
    type_dict = objConfig['config']['model']['type_dict']
    batch_size = objParams['training']['batch_size']
    optimizer = objParams['training']['optimiser']

    info = oscml.data.dataset.get_dataset_info(type_dict)
    node_type_number = len(info.node_types)

    # set model parameters from the config file
    #--------------------------------------
    model_params = {}
    for key, value in objConfig['config']['model']['model_specific'].items():
        model_params.update({key: set_config_param(trial=trial,param_name=key,param=value, all_params=model_params)})

    # constant state vector size throughout the graph convolutional layers
    embedding_dim = model_params['embedding_dim']
    conv_layers = model_params['conv_layers']
    conv_dims = [embedding_dim]*conv_layers
    model_params['conv_dims'] = conv_dims

    # add output dimension to the mlp_units
    model_params['mlp_units'] = model_params.get('mlp_units', []) + [1]

    # add extra params not defined in the config file
    extra_params =  {
        # additional non-hyperparameter values
        'node_type_number': node_type_number, #len(oscml.data.dataset_hopv15.ATOM_TYPES_HOPV15),
        'padding_index': 0,
        'target_mean': transformer.target_mean,
        'target_std': transformer.target_std,
    }
    model_params.update(extra_params)
    logging.info('model params=%s', model_params)

    model_params.pop('mlp_layers',None) # this is not needed for the model creation
    model_params.pop('conv_layers',None) # this is not needed for the model creation

    model = modelCreatorClass(**model_params, optimizer=optimizer)

    return model


def data_preproc(trial, data, objConfig, objParams):

    type_dict = objConfig['config']['model']['type_dict']
    batch_size = objParams['training']['batch_size']

    # datasets
    df_train= data['train']
    df_val= data['val']
    df_test= data['test']
    transformer= data['transformer']

    train_dl, val_dl, test_dl = oscml.models.model_gnn.get_dataloaders(type_dict, df_train, df_val, df_test,
            data['transformer'], batch_size=batch_size)

    processed_data = {
        "train": train_dl,
        "val": val_dl,
        "test": test_dl,
        "transformer": data['transformer']
    }
    return processed_data