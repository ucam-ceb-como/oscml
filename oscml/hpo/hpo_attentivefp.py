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


def create(trial, config, df_train, df_val, df_test, optimizer, transformer, log_dir, freeze=False, fine_tune=False):

    if freeze or fine_tune:
        x_column = config['transfer_learning']['dataset']['x_column'][0]
        y_column = config['transfer_learning']['dataset']['y_column'][0]
    else:
        x_column = config['dataset']['x_column'][0]
        y_column = config['dataset']['y_column'][0]
    featurizer_config = config['model']['featurizer']
    batch_size = config['training']['batch_size']
    dgl_log_dir = log_dir + '/dgl_' + str(trial.number)
        # no standardization of target values supported at the moment
    target_mean = transformer.target_mean
    target_std = transformer.target_std

    if featurizer_config == 'full':
        node_featurizer = dgllife.utils.AttentiveFPAtomFeaturizer()
        edge_featurizer = dgllife.utils.AttentiveFPBondFeaturizer(self_loop=True)
        node_feat_size = node_featurizer.feat_size('h')     # = 39
        edge_feat_size = edge_featurizer.feat_size('e')     # = 11
    else:
        node_featurizer = SimpleAtomFeaturizer2()
        #edge_featurizer = dgllife.utils.AttentiveFPBondFeaturizer(self_loop=True)
        #edge_featurizer = dgllife.utils.CanonicalBondFeaturizer(self_loop=True)
        edge_featurizer = EmptyBondFeaturizes(self_loop=True)
        node_feat_size = node_featurizer.feat_size('h')
        edge_feat_size = edge_featurizer.feat_size('e')


    # set model parameters from the config file
    #--------------------------------------
    model_params = {}
    for key, value in config['model']['model_specific'].items():
        model_params.update({key: set_config_param(trial=trial,param_name=key,param=value, all_params=model_params)})

    # add extra params not defined in the config file
    extra_params = {
        'node_feat_size': node_feat_size,
        'edge_feat_size': edge_feat_size,
        'n_tasks': 1
    }
    model_params.update(extra_params)

    logging.info('model params=%s', model_params)

    if freeze:
        model_predictor = oscml.utils.util_transfer_learning.AttentiveFPTransfer(**model_params)
    else:
        model_predictor = dgllife.model.AttentiveFPPredictor(**model_params)


    dataloader_fct = functools.partial(get_dataloader, smiles_column=x_column, y_column=y_column, transformer=transformer,
        batch_size=batch_size, log_dir=dgl_log_dir, node_featurizer=node_featurizer, edge_featurizer=edge_featurizer,
        collate_fn=collate_molgraphs)

    train_dl = dataloader_fct(df=df_train, file_name="graph_train.bin", shuffle=True)
    val_dl = dataloader_fct(df=df_val, file_name="graph_val.bin", shuffle=False)
    test_dl = dataloader_fct(df=df_test, file_name="graph_test.bin", shuffle=False)

    model = oscml.utils.util_lightning.ModelWrapper(model_predictor, optimizer, target_mean, target_std)

    return model, train_dl, val_dl, test_dl


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
