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

import main
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pipeline.utilities import *

matplotlib.use('Agg')
sns.set(style="ticks", font_scale=1.0)


def plot_pipeline(args, result_keys, result_dict, config_fixed, combined=None):
    figures_path = os.path.join(args.out_dir, 'figures')
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)
    
    for result_key in result_keys:
        if 'keys' in result_key:
            continue
        if 'angle' in result_key:
            xlabel = 'iteration'
        else:
            xlabel = 'epoch'
        ylabel = make_ylabel(result_key)

        result_key_dict = result_dict[result_key]
        
        make_plot(
                result_key_dict=result_key_dict,
                result_key=result_key,
                title=result_key,
                xlabel=xlabel,
                ylabel=ylabel,
                out_dir=figures_path,
                save=True,
                fancyplot=True,
                log_interval=config_fixed['log_interval'],
                combined=combined,
                )


def make_plot(result_key_dict, result_key, title='plot', xlabel='x',
              ylabel=None, out_dir='logs/figures', save=False, show=False,
              fancyplot=True, smooth=30, log_interval=20, no_title=True, combined=False):
    """ Make a plot for each layer, comparing the result_key for all the
    training methods used when creating the result_dict"""
    if ylabel is None:
        ylabel = result_key

    if isinstance(result_key_dict[list(result_key_dict.keys())[0]], np.ndarray):

        fig, ax = plt.subplots()
        for name, array in result_key_dict.items():
            ax.plot(array,'-', label=name)
        
        plt.legend(loc='upper left', frameon=False)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        sns.despine()

        if save:
            file_name = title + '.png'
            plt.savefig(os.path.join(out_dir, file_name))
            plt.close()
        if show:
            plt.show()

    if isinstance(result_key_dict[list(result_key_dict.keys())[0]], pd.DataFrame):
        if fancyplot and ("angle" or "condition") in result_key:
            try:
                plot_rolling(result_key_dict=result_key_dict,
                            result_key=result_key,
                            title=result_key,
                            xlabel=xlabel,
                            ylabel=ylabel,
                            out_dir=out_dir,
                            save=True,
                            show=show,
                            smooth=smooth,
                            log_interval=log_interval,
                            no_title=no_title,
                            combined=combined)
            except Exception as error:
                print('plot_unrolling failed for {}'.format(result_key))
                print(error)


def make_plot_smooth(result_key_dict, result_key, title='plot', xlabel='x',
              ylabel=None, out_dir='logs/figures', save=False,  show=False, smooth=30, fancyplot=True):

    if ylabel is None:
        ylabel = result_key

    if isinstance(result_key_dict[list(result_key_dict.keys())[0]], np.ndarray):
        plt.figure()
        legend = []
        for name, array in result_key_dict.items():
            legend.append(name)
            plt.plot(array)
        plt.legend(legend, loc='upper left', frameon=False)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        sns.despine()

        if save:
            file_name = title + '.png'
            plt.savefig(os.path.join(out_dir, file_name))
        if show:
            plt.show()

    if isinstance(result_key_dict[list(result_key_dict.keys())[0]], pd.DataFrame):
        layerwise_dict, legend = create_layerwise_result_dict(result_key_dict)
        for idx, df in layerwise_dict.items():

            df.to_pickle(os.path.join(f"df{idx}.pkl"))
            df = pd.read_pickle(os.path.join(f"df{idx}.pkl"))

            ncols = len(df.columns)
            indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=smooth)
            print('forward indexer')
            for col in range(ncols):
                df[f'roll_mean{col}'] = df.iloc[:, col].rolling(window=indexer).mean()
            for col in range(ncols):
                df[f'roll_std{col}'] = df.iloc[:, col].rolling(window=indexer).std()


            df_points = df.iloc[:, :ncols]
            df_roll_mean = df.iloc[:, ncols:2*ncols].T.reset_index(drop=True).T
            df_roll_std = df.iloc[:, 2*ncols:].T.reset_index(drop=True).T

            df_upper = df_roll_mean.add(df_roll_std, fill_value=0)
            df_lower = df_roll_mean.sub(df_roll_std, fill_value=0)

            plt.figure()
            figure_title = title + ' layer ' + str(idx+1)
            ax = df_points.plot(title=figure_title, style='.', colormap='viridis')
            df_roll_mean.plot(title=figure_title, style='-', ax=ax, colormap='viridis')
            df_upper.plot(title=figure_title, style='--', ax=ax, colormap='viridis')
            df_lower.plot(title=figure_title, style='--', ax=ax, colormap='viridis')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            plt.legend(legend, loc='upper left', frameon=False)
            exit()

        if save:
            file_name = title + '_layer' + str(idx) + '.png'
            plt.savefig(os.path.join(out_dir, file_name))
        if show:
            plt.show()


def plot_rolling(result_key_dict, result_key, title='plot', xlabel='x',
                 ylabel=None, out_dir='logs/figures', save=False, show=False,
                 smooth=30, subplots=True, save_individual_plots=False, omit_last=False, log_interval=20,
                 no_title=True, combined=False):
    
    experiments = []
    max_idx = 0

    for name, df in result_key_dict.items():
        nlayers = len(df.columns)
        for layer in range(nlayers):
            df[f'roll_mean_{layer}'] = df.iloc[:, layer].rolling(window=indexer, min_periods=1).mean()
            df[f'roll_std_{layer}'] = df.iloc[:, layer].rolling(window=indexer, min_periods=1).std()
            df[f"lower{layer}"] = df[f"roll_mean_{layer}"].sub(df[f"roll_std_{layer}"], fill_value=0)
            df[f"upper{layer}"] = df[f"roll_mean_{layer}"].add(df[f"roll_std_{layer}"], fill_value=0)
        df['batch_idx'] = df.index * log_interval
        df['experiment'] = name
        experiments.append(name)
        if max_idx < df['batch_idx'].max():
            max_idx = df['batch_idx'].max()
    df = pd.concat(result_key_dict.values(), ignore_index=True)
    color_palette = sns.color_palette("Set1", len(experiments))
    if len(experiments) >= 6:
        color_palette[5] = (0.33203125, 0.41796875, 0.18359375)
    if len(experiments) >= 9:
        color_palette[8] = (0., 0.796875, 0.796875)
    sns.set_palette(color_palette)
    
    if omit_last:
        nlayers = nlayers-1
    plt.close('all')
    
    max_value = 0
    for layer in range(nlayers):
        ax = sns.lineplot(x="batch_idx", y=f"roll_mean_{layer}", hue="experiment", data=df, estimator=None)
        for i, experiment in enumerate(experiments):
            df_fill = df.loc[df["experiment"]==experiment]
            ax.fill_between(df_fill["batch_idx"], df_fill[f"lower{layer}"], df_fill[f"upper{layer}"], alpha=0.3, lw=0, color=color_palette[i])
            if max_value < df_fill[[f"lower{layer}", f"upper{layer}"]].max().max():
                max_value = df_fill[[f"lower{layer}", f"upper{layer}"]].max().max()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim([log_interval, max_idx])
        ax.legend(loc='upper left', frameon=False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        sns.despine()
        if not no_title:
            plt.title(f"Layer {layer+1}: {result_key}")
        if save_individual_plots:
            file_name = title + '_layer' + str(layer) + '.png'
            plt.savefig(os.path.join(out_dir, file_name), dpi=300)
        if show:
            plt.show()
        else:
            plt.close()
    
    if subplots:
        fig, axes = plt.subplots(1, nlayers, figsize=(nlayers*4 + 4, 5))
        plt.tight_layout()
        for layer in range(nlayers):
            if nlayers==1:
                ax = axes
            else:
                ax = axes[layer]
            sns.lineplot(x="batch_idx", y=f"roll_mean_{layer}", hue="experiment", data=df, estimator=None, ax=ax)
            for i, experiment in enumerate(experiments):
                df_fill = df.loc[df["experiment"] == experiment]
                ax.fill_between(df_fill["batch_idx"], df_fill[f"lower{layer}"], df_fill[f"upper{layer}"], alpha=0.3, lw=0,
                                color=color_palette[i])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_ylim([-0.005, max_value])
            ax.set_xlim([0, max_idx])
            ax.legend_.remove()
            if layer != 0:
                ax.set_ylabel("")
            if 'condition' not in result_key and 'jac' not in result_key:
                ax.set_title(f"Layer {layer+1}")
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            sns.despine()

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles=handles[1:], labels=labels[1:], loc="center right", frameon=False)
        plt.tight_layout()
        if 'condition' in result_key or 'jac' in result_key:
            plt.subplots_adjust(right=0.80)
        elif 'fixed' in labels:
            plt.subplots_adjust(right=0.90)
        else:
            plt.subplots_adjust(right=0.90)

        if save:
            file_name = title + '_subplots.png'
            plt.savefig(os.path.join(out_dir, file_name))
        if show:
            plt.show()
        else:
            plt.close()

def make_ylabel(result_key):
    if result_key == 'gnt_angles':
        ylabel = r'$\Delta W_i \angle \Delta W_i^{MN}$ [$^\circ$]'
    elif result_key == 'bp_angles':
        ylabel = r'$\Delta W_i \angle \Delta W_i^{BP}$ [$^\circ$]'
    elif result_key == 'gn_angles':
        ylabel = r'$\Delta W_i \angle \Delta W_i^{GN}$ [$^\circ$]'
    elif result_key == 'ndi_angles':
        ylabel = r'$\Delta W_i \angle \Delta W_i^{NDI}$ [$^\circ$]'
    elif result_key == 'condition_gn_angles':
        ylabel = r'$||P_{\bar{J}^{T}}\bar{Q}||_{F} / ||\bar{Q}||_{F}$ [$^\circ$]'
    elif result_key == 'condition_gn_angles_init':
        ylabel = r'$||P_{\bar{J}^{T}}\bar{Q}_{init}||_{F} / ||\bar{Q}_{init}||_{F}$ [$^\circ$]'
    elif result_key == 'jac_transpose_angles':
        ylabel = r'$\bar{Q} \angle \bar{J}^{T}$ [$^\circ$]'
    elif result_key == 'jac_transpose_angles_init':
        ylabel = r'$\bar{Q}_{init} \angle \bar{J}^{T}$ [$^\circ$]'
    elif result_key == 'jac_pinv_angles':
        ylabel = r'$\bar{Q} \angle \bar{J}^{T}(\bar{J}\bar{J}^{T}+\gamma I)^{-1}$ [$^\circ$]'
    elif result_key == 'jac_pinv_angles_init':
        ylabel = r'$\bar{Q}_{init} \angle \bar{J}^{T}(\bar{J}\bar{J}^{T}+\gamma I)^{-1}$ [$^\circ$]'
    elif result_key == 'loss_test' or result_key == 'loss_train':
        ylabel = 'loss'
    elif result_key == 'nullspace_relative_norm_angles':
        ylabel = r'$||\Delta W^{null}||/||\Delta W||$'
    elif result_key == 'converged' or result_key == 'not_converged' or result_key == 'diverged':
        ylabel = 'number of samples'
    else:
        ylabel = r'$\angle$ [$^\circ$]'
    return ylabel