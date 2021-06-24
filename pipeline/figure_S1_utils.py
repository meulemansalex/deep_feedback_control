#!/usr/bin/env python3
# Copyright 2021 Alexander Meulemans, Matilde Tristany Farinha, Javier Garcia Gordonez
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script created figure S1.
* Maximum eigenvalue of different dynamics matrices A, over the course of training
* Alignment with GN updates
* Alignment with GNT updates (should be close to GN updates)
* GN condition
"""


import os
import argparse
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pipeline.figure_utils import make_ylabel, plot_rolling


def run(args=None):
    if not args:
        parser = argparse.ArgumentParser()
        parser.add_argument('--out_dir', type=str, default='logs/toy_experiments/fig3',
                            help='Directory where the results were saved.')
        parser.add_argument('--toy_experiment_number', type=int, default=1,
                            help='Different toy experiments could be carried out, and'
                                 'figures to be generated are different. Please specify.')
        parser.add_argument('--individual_plots', action='False',
                            help='Flag indicating whether individual plots for each layer and'
                                 'stability matrix A should be produced. Otherwise, subplots will be'
                                 'used to display all necessary information in a single figure.')
        args = parser.parse_args()

    filename = os.path.join(args.out_dir, 'result_dict.pickle')
    with open(filename, 'rb') as f:
        result_dict = pickle.load(f)

    result_keys = [k for k in result_dict.keys()]
    names = [k for k in result_dict[result_keys[0]].keys()]
    name = names[-1]

    our_color_palette = sns.color_palette("Set1", len(names))
    if len(names) >= 6:
        our_color_palette[5] = (0.33203125, 0.41796875, 0.18359375)
    if len(names) >= 9:
        our_color_palette[8] = (0., 0.796875, 0.796875)
    sns.set_palette(our_color_palette)

    if "log_interval" in result_dict:
        log_interval = result_dict['log_interval'][name]
    else:
        print('Log interval not specified. Set to 10 by default.')
        log_interval = 10

    figures_path = os.path.join(args.out_dir, 'figS1')
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)

    plot_keys = result_dict['max_eig_keys'][name]
    dict_key = "max_eig"
    plot_mean_std(result_dict, plot_keys, dict_key, names, figures_path, log_interval, color_palette=our_color_palette, ylabel=r'$\lambda_{max}$')

    dict_key = "max_eig_bcn"
    plot_mean_std(result_dict, plot_keys, dict_key, names, figures_path, log_interval, color_palette=our_color_palette, ylabel=r'$\lambda_{max}$')

    plot_keys = result_dict['norm_r_keys'][name]
    dict_key = "norm_r"
    plot_mean_std(result_dict, plot_keys, dict_key, names, figures_path, log_interval, color_palette=our_color_palette, ylabel='norm($||r_i||$) - mean($||r_i||$) / mean($||r_i||$)')

    keys = ['gnt_angles', 'gn_angles','condition_gn_angles', 'nullspace_relative_norm_angles']
    titles = ["Angle GNT", "Angle GN", "Condition GN", "Nullspace norm"]
    subplots = [True, True, False, False]

    for i, k in enumerate(keys):
        ylabel = make_ylabel(k)
        plot_rolling(result_key_dict=result_dict[k],
                    result_key=k,
                    title=titles[i],
                    xlabel='Iteration',
                    ylabel=ylabel,
                    out_dir=figures_path,
                    save=True,
                    show=False,
                    omit_last=False,
                    subplots=subplots[i],
                    smooth=5*log_interval,
                    log_interval=log_interval,
                    no_title=False)

print('Toy experiment figures created!')


def plot_mean_std(result_dict, plot_keys, dict_key, names, figures_path, log_interval=10,
                  subplots=True, color_palette=None, ylabel='Maximum eigenvalue'):
    """
    Plots of max eigenvalues / mod r, along training, for the different A matrices or diff layers
    :param result_dict: dictionary with all the results
    :param plot_keys: keys for which a subplot / individual plot will be generated
    :param dict_key: key to which adding "_mean" or "_std" will give the final key to access `result_dict`
    :param names: names of the different experiments (different runs) to plot together
    :param figures_path: path to which figures will be saved
    :param log_interval: step for the x axis
    :param color_palette: color palette, to be consistent with other generated plots
    :return: Nothing (figures are saved in `figures_path`)
    """

    if subplots:
        if len(plot_keys) == 4: fig, axes = plt.subplots(2,2,figsize=(13, 10))
        else: fig, axes = plt.subplots(len(plot_keys), 1, figsize=(len(plot_keys)*4 + 4, 5))
        axes = axes.reshape(-1,)
        for i, k in enumerate(plot_keys):
            axes[i].set_title(k)
            for n, name in enumerate(names):
                mean = []
                std = []
                for j in range(i, len(result_dict[dict_key+'_mean'][name]), len(plot_keys)):
                    mean.append(result_dict[dict_key+'_mean'][name][j])
                    std.append(result_dict[dict_key+'_std'][name][j])
                mean = np.asarray(mean)
                std = np.asarray(std)
                batch_idx = np.arange(len(mean)) * log_interval
                axes[i].plot(batch_idx, mean, label=name)
                axes[i].fill_between(batch_idx, mean - std, mean + std, lw=0, alpha=0.3, color=color_palette[n])

            axes[i].legend(loc='upper left')
            axes[i].set_xlabel('Iteration')
            axes[i].set_ylabel(ylabel)

            xlim = axes[i].get_xlim()
            axes[i].hlines(0, xlim[0], xlim[1], colors='k', lw=10, alpha=0.8)
            axes[i].set_xlim(xlim)

        plt.savefig(os.path.join(figures_path, dict_key))
        plt.close()

    else:
        for i, k in enumerate(plot_keys):
            plt.figure(figsize=(13, 10))
            plt.title(k)
            for n, name in enumerate(names):
                mean = []
                std = []
                for j in range(i, len(result_dict[dict_key+'_mean'][name]), len(plot_keys)):
                    mean.append(result_dict[dict_key+'_mean'][name][j])
                    std.append(result_dict[dict_key+'_std'][name][j])
                mean = np.asarray(mean)
                std = np.asarray(std)
                batch_idx = np.arange(len(mean)) * log_interval
                plt.plot(batch_idx, mean, label=name)
                plt.fill_between(batch_idx, mean - std, mean + std, lw=0, alpha=0.3, color=color_palette[n])

            plt.legend(loc='upper left')
            plt.xlabel('Iteration')
            plt.ylabel('Maximum eigenvalue')

            xlim = plt.xlim()
            plt.hlines(0, xlim[0], xlim[1], colors='k', lw=10, alpha=0.8)
            plt.xlim(xlim)

            plt.savefig(os.path.join(figures_path, k))
            plt.close()


if __name__ == '__main__':
    run()