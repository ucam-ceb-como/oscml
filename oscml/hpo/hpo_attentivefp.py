import functools
import logging

import dgl
import dgllife.data
import dgllife.model
import dgllife.utils
import torch
import torch.utils.data

import oscml.utils.util_lightning


class AttentiveFPWrapper(oscml.utils.util_lightning.OscmlModule, dgllife.model.AttentiveFPPredictor):

    def __init__(self, node_feat_size, edge_feat_size, num_layers=2, num_timesteps=2, graph_feat_size=200, n_tasks=1, dropout=0.,  optimizer=None):

        oscml.utils.util_lightning.OscmlModule.__init__(self, optimizer, 1, 0) #target_mean, target_std)
        dgllife.model.AttentiveFPPredictor.__init__(self, node_feat_size, edge_feat_size, num_layers, num_timesteps, graph_feat_size, n_tasks, dropout)
        #TODO AE URGENT
        target_mean = 4.126659584567247
        target_std= 2.407382175602577

        logging.info('initializing ' + str(locals()))

    def forward(self, *args, **kwargs):

        args = args[0]
        #print(type(args))
        #print(args)

        return super(*args, **kwargs)
    

def create(trial, config, df_train, df_val, df_test, optimizer):

    model = dgllife.model.AttentiveFPPredictor(
        node_feat_size=39,      #exp_configure['in_node_feats'],
        edge_feat_size=11,      #exp_configure['in_edge_feats'],
        num_layers=4,           #exp_configure['num_layers'],
        num_timesteps=2,        #exp_configure['num_timesteps'],
        graph_feat_size=200,    #exp_configure['graph_feat_size'],
        dropout=0.,             #exp_configure['dropout'],
        n_tasks=1               #exp_configure['n_tasks']
    )

    #TODO AE URGENT target_mean and std.
    #target_mean = transformer.target_mean   #4.126659584567247
    #target_std= transformer.target_std      #2.407382175602577
    target_mean = 0
    target_std = 1
    model = oscml.utils.util_lightning.ModelWrapper(model, optimizer, target_mean, target_std)


    """
    model = AttentiveFPWrapper(
                node_feat_size=39,      #exp_configure['in_node_feats'],
                edge_feat_size=11,      #exp_configure['in_edge_feats'],
                num_layers=4,           #exp_configure['num_layers'],
                num_timesteps=2,        #exp_configure['num_timesteps'],
                graph_feat_size=200,    #exp_configure['graph_feat_size'],
                dropout=0.,             #exp_configure['dropout'],
                n_tasks=1,              #exp_configure['n_tasks']
                optimizer=optimizer
            )
    """


    train_dl, val_dl, test_dl = oscml.hpo.hpo_attentivefp.get_dataloaders(df_train, df_val, df_test, column_smiles='SMILES_str', column_target='pce', batch_size=250)

    return model, train_dl, val_dl, test_dl

def get_dataloaders(df_train, df_val, df_test, column_smiles, column_target, batch_size):

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
                                 #TODO AE MED change default path
                                 cache_file_path='dgl_regression_results' + '/graph_train.bin',
                                 task_names=[column_target],
                                 n_jobs=0)

    dl_train = torch.utils.data.DataLoader(dataset=dgl_ds_train, batch_size=batch_size, shuffle=True,
                                collate_fn=collate_molgraphs, num_workers=0)
    
    dgl_ds_val = dgllife.data.MoleculeCSVDataset(df=df_val,
                                 smiles_to_graph=smiles_to_graph,
                                 node_featurizer=node_featurizer,
                                 edge_featurizer=edge_featurizer,
                                 smiles_column=column_smiles,
                                 #TODO AE MED change default path
                                 cache_file_path='dgl_regression_results' + '/graph_val.bin',
                                 task_names=[column_target],
                                 n_jobs=1)

    dl_val = torch.utils.data.DataLoader(dataset=dgl_ds_val, batch_size=batch_size, shuffle=False,
                                collate_fn=collate_molgraphs, num_workers=1)


    #TODO AE MED return dataloaders for val and test
    return dl_train, dl_val, None

#TODO AE LOW copyright 
# This method was copied from dlg /csv_data_configuration/utils.py
# and extended with the last if block of predict method - also from /csv_data_configuration/utils.py
# plus adaption to implementation of method training_step in util_lightning 
def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.

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
    #TODO AE MID Device cpu gpu 
    node_feats = bg.ndata.pop('h')  # .to(args['device'])
    edge_feats = bg.edata.pop('e')  # .to(args['device'])

    x = [bg, node_feats, edge_feats]

    #TODO AE MID use mask to get the labels
    # Here, we only use one label value, so we don't the mask
    # run_a_train_epoch() in regression_train.py 
    # labels, masks = labels.to(args['device']), masks.to(args['device'])
    y = labels #torch.flatten(labels)

    return x, y
