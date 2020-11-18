import functools
import logging

import dgl
import dgllife.data
import dgllife.model
import dgllife.utils
import torch
import torch.utils.data

import oscml.utils.util_lightning


def create(trial, config, df_train, df_val, df_test, optimizer, dataset, log_dir):

    # dataloaders
    info = oscml.data.dataset.get_dataset_info(dataset)
    column_smiles = info.column_smiles
    column_target = info.column_target
    dgl_log_dir = log_dir + '/dgl_' + str(trial.number)
    train_dl, val_dl, test_dl = oscml.hpo.hpo_attentivefp.get_dataloaders(df_train, df_val, df_test, 
            column_smiles=column_smiles, column_target=column_target, batch_size=250, log_dir=dgl_log_dir)

    # define models and params
    model_params =  {
        'node_feat_size': 39,
        'edge_feat_size': 11,
        'graph_feat_size': trial.suggest_int('graph_feat_size', 16, 256),
        'num_layers': trial.suggest_int('num_layers', 1, 6),
        'num_timesteps': trial.suggest_int('num_timesteps', 1, 4),
        'dropout': trial.suggest_uniform('dropout', 0., 0.2),
        'n_tasks': 1
    }

    logging.info('model params=%s', model_params)

    model_predictor = dgllife.model.AttentiveFPPredictor(**model_params)

    # no standardization of target values supported at the moment
    target_mean = 0
    target_std = 1
    model = oscml.utils.util_lightning.ModelWrapper(model_predictor, optimizer, target_mean, target_std)

    return model, train_dl, val_dl, test_dl


def get_dataloaders(df_train, df_val, df_test, column_smiles, column_target, batch_size, log_dir):

    #TODO AE URGENT for development only
    df_train = df_train[:150]
    df_val = df_val[:50]
    df_test = df_test[:50]


    node_featurizer = dgllife.utils.AttentiveFPAtomFeaturizer()
    edge_featurizer = dgllife.utils.AttentiveFPBondFeaturizer(self_loop=True)

    smiles_to_graph=functools.partial(dgllife.utils.smiles_to_bigraph, add_self_loop=True)
    dgl_ds_train = dgllife.data.MoleculeCSVDataset(df=df_train,
                                 smiles_to_graph=smiles_to_graph,
                                 node_featurizer=node_featurizer,
                                 edge_featurizer=edge_featurizer,
                                 smiles_column=column_smiles,
                                 cache_file_path=log_dir + '/graph_train.bin',
                                 task_names=[column_target],
                                 n_jobs=0)

    dl_train = torch.utils.data.DataLoader(dataset=dgl_ds_train, batch_size=batch_size, shuffle=True,
                                collate_fn=collate_molgraphs, num_workers=0)
    
    dgl_ds_val = dgllife.data.MoleculeCSVDataset(df=df_val,
                                 smiles_to_graph=smiles_to_graph,
                                 node_featurizer=node_featurizer,
                                 edge_featurizer=edge_featurizer,
                                 smiles_column=column_smiles,
                                 cache_file_path=log_dir + '/graph_val.bin',
                                 task_names=[column_target],
                                 n_jobs=0)

    dl_val = torch.utils.data.DataLoader(dataset=dgl_ds_val, batch_size=batch_size, shuffle=False,
                                collate_fn=collate_molgraphs, num_workers=0)

    dgl_ds_test = dgllife.data.MoleculeCSVDataset(df=df_test,
                                 smiles_to_graph=smiles_to_graph,
                                 node_featurizer=node_featurizer,
                                 edge_featurizer=edge_featurizer,
                                 smiles_column=column_smiles,
                                 cache_file_path=log_dir + '/graph_test.bin',
                                 task_names=[column_target],
                                 n_jobs=0)

    dl_test = torch.utils.data.DataLoader(dataset=dgl_ds_test, batch_size=batch_size, shuffle=False,
                                collate_fn=collate_molgraphs, num_workers=0)

    return dl_train, dl_val, dl_test


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
