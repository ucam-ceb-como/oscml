import functools
import logging
import dgl
import dgllife.data
import dgllife.model
import dgllife.utils
import torch
import torch.utils.data
from oscml.utils.util_config import set_config_param
import oscml.utils.util_lightning
import oscml.utils.util_transfer_learning
from oscml.hpo.objclass import Objective
from oscml.hpo.hpo_utils import preproc_training_params
from oscml.utils.util_transfer_learning import AttentiveFPTransfer
from dgllife.model import AttentiveFPPredictor
from oscml.hpo.hpo_utils import NN_model_train, NN_addBestModelRetrainCallback, NN_valDataCheck
from oscml.hpo.hpo_utils import NN_model_train_cross_validate, NN_logBestTrialRetraining, NN_transferLearningCallback
from oscml.hpo.hpo_utils import NN_prepareTransferLearningModel, NN_logTransferLearning


def getObjectiveAttentiveFP(modelName, data, config, logFile, logDir,
                            crossValidation, bestTrialRetraining=False, transferLearning=False):

    if not crossValidation:
        # for not cv job, make sure there is non empty validation set
        # as NN methods require it for training
        data = NN_valDataCheck(data, config, transferLearning)

    objectiveAttentiveFP = Objective(modelName=modelName, data=data, config=config,
                        logFile=logFile, logDir=logDir)

    if transferLearning and config['transfer_learning']['freeze_and_train']:
        modelCreatorClass = AttentiveFPTransfer
    else:
        modelCreatorClass = AttentiveFPPredictor

    model_trainer_func = NN_model_train_cross_validate if crossValidation else NN_model_train

    objectiveAttentiveFP.addPreModelCreateTask(objParamsKey='training', funcHandle=preproc_training_params)
    objectiveAttentiveFP.addPreModelCreateTask(objParamsKey='featurizer', funcHandle=get_featuriser)
    objectiveAttentiveFP.setModelCreator(funcHandle=model_create, extArgs=[modelCreatorClass])
    objectiveAttentiveFP.setModelTrainer(funcHandle=model_trainer_func, extArgs=[data_preproc])

    if bestTrialRetraining:
        objectiveAttentiveFP.addPostModelCreateTask(objParamsKey='callbackBestTrialRetraining', funcHandle=NN_addBestModelRetrainCallback)
        objectiveAttentiveFP.addPostTrainingTask(objParamsKey='logBestTrialRetrain', funcHandle=NN_logBestTrialRetraining)

    if transferLearning:
        objectiveAttentiveFP.addPostModelCreateTask(objParamsKey='callbackTransferLearning', funcHandle=NN_transferLearningCallback)
        objectiveAttentiveFP.addPostModelCreateTask(objParamsKey='transferLearningModel', funcHandle=NN_prepareTransferLearningModel)
        objectiveAttentiveFP.addPostTrainingTask(objParamsKey='logTransferLearning', funcHandle=NN_logTransferLearning)

    return objectiveAttentiveFP

class AttentiveFPObjective(Objective):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, trial):
        self.objconfig['training'] = preproc_training_params(trial, self.objconfig)
        self.objconfig['featurizer'] = get_featuriser(self.objconfig)

        model = model_create(trial, self.model_creator, self.objconfig, self.data['transformer'])

        self.obj_value = self.model_train(trial, model, self.data, self.objconfig, data_preproc)
        return self.obj_value

def get_featuriser(trial, data, objConfig, objParams):
    featurizer = {
        "node_featurizer": None,
        "edge_featurizer": None,
        "node_feat_size": None,
        "edge_feat_size": None
    }
    featurizer_config = objConfig['config']['model']['featurizer']

    if featurizer_config == 'full':
        featurizer['node_featurizer'] = dgllife.utils.AttentiveFPAtomFeaturizer()
        featurizer['edge_featurizer'] = dgllife.utils.AttentiveFPBondFeaturizer(self_loop=True)
        featurizer['node_feat_size'] = featurizer['node_featurizer'].feat_size('h')     # = 39
        featurizer['edge_feat_size'] = featurizer['edge_featurizer'].feat_size('e')     # = 11
    else:
        featurizer['node_featurizer'] = SimpleAtomFeaturizer2()
        #edge_featurizer = dgllife.utils.AttentiveFPBondFeaturizer(self_loop=True)
        #edge_featurizer = dgllife.utils.CanonicalBondFeaturizer(self_loop=True)
        featurizer['edge_featurizer'] = EmptyBondFeaturizes(self_loop=True)
        featurizer['node_feat_size'] = featurizer['node_featurizer'].feat_size('h')
        featurizer['edge_feat_size'] = featurizer['edge_featurizer'].feat_size('e')
    return featurizer

def model_create(trial, data, objConfig, objParams, modelCreatorClass):

    transformer = data['transformer']
    optimizer = objParams['training']['optimiser']
    featurizer = objParams['featurizer']
    # no standardization of target values supported at the moment
    target_mean = transformer.target_mean
    target_std = transformer.target_std

    # set model parameters from the config file
    #--------------------------------------
    model_params = {}
    for key, value in objConfig['config']['model']['model_specific'].items():
        model_params.update({key: set_config_param(trial=trial,param_name=key,param=value, all_params=model_params)})

    # add extra params not defined in the config file
    extra_params = {
        'node_feat_size': featurizer['node_feat_size'],
        'edge_feat_size': featurizer['edge_feat_size'],
        'n_tasks': 1
    }
    model_params.update(extra_params)

    logging.info('model params=%s', model_params)

    model_predictor = modelCreatorClass(**model_params)
    model = oscml.utils.util_lightning.ModelWrapper(model_predictor, optimizer, target_mean, target_std)

    return model
class SimpleAtomFeaturizer(dgllife.utils.BaseAtomFeaturizer):

    def __init__(self, atom_data_field='h'):
        super().__init__(
            featurizer_funcs={atom_data_field: dgllife.utils.ConcatFeaturizer(
                [dgllife.utils.atom_type_one_hot,
                 dgllife.utils.atom_is_aromatic]
            )})

# TODO AE URGENT check atomtypes for Osaka, may remove unused types
class SimpleAtomFeaturizer2(dgllife.utils.BaseAtomFeaturizer):

    def __init__(self, atom_data_field='h'):
        super().__init__(
            featurizer_funcs={atom_data_field: dgllife.utils.ConcatFeaturizer(
                [functools.partial(dgllife.utils.atom_type_one_hot, allowable_set=[
                    'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'H',
                    'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At'], encode_unknown=True),
                 dgllife.utils.atom_is_aromatic]
            )})

# TODO AE URGENT: still edge feature size = 1 not = 0!!!
class EmptyBondFeaturizes(dgllife.utils.BaseBondFeaturizer):
    def __init__(self, bond_data_field='e', self_loop=False):
        super().__init__(
            featurizer_funcs={bond_data_field: dgllife.utils.ConcatFeaturizer(
                #[dgllife.utils.bond_type_one_hot]
                []
            )}, self_loop=self_loop)
            #featurizer_funcs={}, self_loop=self_loop)

def get_dataloader(df, file_name, shuffle, smiles_column, y_column, transformer, batch_size, log_dir, node_featurizer, edge_featurizer, collate_fn):

    #TODO AE URGENT for development only
    #df = df[:150]

    smiles_to_graph=functools.partial(dgllife.utils.smiles_to_bigraph, add_self_loop=True)

    transformed_y = transformer.transform(df)
    transformed_column_name = y_column + "_transformed"
    df[transformed_column_name] = transformed_y

    dataset = dgllife.data.MoleculeCSVDataset(df=df,
                                 smiles_to_graph=smiles_to_graph,
                                 node_featurizer=node_featurizer,
                                 edge_featurizer=edge_featurizer,
                                 smiles_column=smiles_column,
                                 cache_file_path=log_dir + '/' + file_name,
                                 task_names=[transformed_column_name],
                                 n_jobs=0)

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=0)
    return dataloader

def collate_molgraphs(data):
    # This method was copied from dlg /csv_data_configuration/utils.py and adapted
    """
    Batching a list of datapoints for dataloader.

    Parameters
    ----------
    data : list of 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and a binary
        mask indicating the existence of labels.

    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels.
    """
    smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)

    #return smiles, bg, labels, masks
    #TODO AE MED Device cpu gpu
    node_feats = bg.ndata.pop('h')  # .to(args['device'])
    edge_feats = bg.edata.pop('e')  # .to(args['device'])

    x = [bg, node_feats, edge_feats]

    # Here, we only use one label value, so we don't the mask
    # run_a_train_epoch() in regression_train.py
    # labels, masks = labels.to(args['device']), masks.to(args['device'])
    y = labels

    return x, y

def data_preproc(trial, data, objConfig, objParams):
    x_column = objConfig['config']['dataset']['x_column'][0]
    y_column = objConfig['config']['dataset']['y_column'][0]
    featurizer = objParams['featurizer']
    batch_size = objParams['training']['batch_size']
    transformer = data['transformer']
    dgl_log_dir = objConfig['log_dir'] + '/dgl_' + str(trial.number)

    dataloader_fct = functools.partial(get_dataloader, smiles_column=x_column, y_column=y_column, transformer=transformer,
        batch_size=batch_size, log_dir=dgl_log_dir, node_featurizer=featurizer['node_featurizer'],
        edge_featurizer=featurizer['edge_featurizer'], collate_fn=collate_molgraphs)

    train_dl = dataloader_fct(df=data['train'], file_name="graph_train.bin", shuffle=True)
    val_dl = dataloader_fct(df=data['val'], file_name="graph_val.bin", shuffle=False)
    test_dl = dataloader_fct(df=data['test'], file_name="graph_test.bin", shuffle=False)

    processed_data = {
        "train": train_dl,
        "val": val_dl,
        "test": test_dl,
        "transformer": data['transformer']
    }
    return processed_data