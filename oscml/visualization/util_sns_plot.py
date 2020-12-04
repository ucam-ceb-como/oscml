import argparse

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def contour_plot(hpo_result):
    df_hpo_result = pd.read_csv(hpo_result, index_col=[0])
    # TODO finish the plot for final prediction results

    df_hpo_result = df_hpo_result.drop(columns=['number', 'datetime_start', 'datetime_complete', 'duration', 'state'])
    df_hpo_result = df_hpo_result.replace(np.nan, 0)

    sns_pair_plot = sns.pairplot(df_hpo_result)
    sns_pair_plot.savefig('test_pair_plot.svg')

    sns_pair_grid = sns.PairGrid(df_hpo_result)
    sns_pair_grid.map_diag(sns.kdeplot)
    sns_pair_grid.map_offdiag(sns.kdeplot, n_levels=6)
    sns_pair_grid.savefig('test_pair_grid.svg')


if __name__ == '__main__':
    #contour_plot('./best_trial_98_svr_hpo_result_trials_part2.csv')
    contour_plot('./logs/hpo_20201203_232802/hpo_result.csv')

# def init_plot():
#     sns.set(font_scale=1)
#     sns.set_style('whitegrid',
#                   {'axes.grid': True,
#                    'grid.linestyle': u'--',
#                    'axes.edgecolor': '0.1',
#                    'axes.labelcolor': '0',
#                    'axes.labelsize': 15,
#                    'axes.titlesize': 15,
#                    'legend.fontsize': 15,
#                    'xtick.labelsize': 15,
#                    'ytick.labelsize': 15,
#                    })
#     plt.rcParams["patch.force_edgecolor"] = True
#
#
# def retrieve_prediction(ckpt_path, model, trainer, train_dl, val_dl, test_dl):
#     model.load_state_dict(torch.load(ckpt_path)['state_dict'])
#     model.eval()
#     training_result = trainer.test(model, test_dataloaders=train_dl)[0]
#     val_result = trainer.test(model, test_dataloaders=val_dl)[0]
#     test_result = trainer.test(model, test_dataloaders=test_dl)[0]
#
#     return training_result, val_result, test_result
#
#
# def plot_prediction():
#     init_plot()
#
#
#
# data_train=pd.DataFrame(predictions_train)
#
# g_train = sns.jointplot(x="Train", y="Train_pred", data=data_train,
#              kind='reg', scatter_kws={'alpha':0.5, 's':20}, height=5)
#
# g_train.ax_joint.set_xlabel('Measured Permeability')
# g_train.ax_joint.set_ylabel('Predicted Permeability')
# text = "R$^2$ = {:0.2} (Training)".format(r2_train)
# plt.annotate(text,
#              xy=(0.1, 0.95),
#              xycoords='axes fraction')
# plt.savefig(figures_dir+"{0}_train_reg.svg".format(model_name))
#
#
# data_test=pd.DataFrame(predictions_test)
#
# g_test = sns.jointplot(x="Test", y="Test_pred", data=data_test,
#              kind='reg', scatter_kws={'alpha':0.5, 's':20}, height=5)
#
# g_test.ax_joint.set_xlabel('Measured Permeability')
# g_test.ax_joint.set_ylabel('Predicted Permeability')
# text = "R$^2$ = {:0.2} (Testset)".format(r2_test)
# plt.annotate(text,
#              xy=(0.1, 0.95),
#              xycoords='axes fraction')
# plt.savefig(figures_dir+"{0}_test_reg.svg".format(model_name))
