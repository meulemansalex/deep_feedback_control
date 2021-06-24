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

import os
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib
import matplotlib.ticker as ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

out_dir = "./logs/toy_experiments/fig3"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
linear = False

experiment_dirs = []
for entry in os.scandir(out_dir):
    if entry.is_dir(): experiment_dirs.append(entry)

new_keys = {
    'gn_angles_network': 'gn_angles',
    'gnt_angles_network': 'mn_angles',
    'ndi_angles_network': 'ssa_angles',
    'dev_r_mean': 'condition_1',
    'condition_gn_angles': 'condition_2',
}

toy_exp_df = dict()
for k in new_keys.keys():
    toy_exp_df[new_keys[k]] = dict()
toy_exp_df['eig_A'] = dict()
toy_exp_df['eig_A_inst'] = dict()

fb_type_keys = {
    'TrainFB_RandomInit_Pretrain':'TrainFB',
    'FreezeFB_WeightProduct':'FreezeFB',
}

for experiment in experiment_dirs:
    dir = os.path.join(experiment.path, 'result_dict')
    if 'dfc_ssa' in experiment.name.lower():
        max_eig_key = 'max_eig_mean'
        for fb_type in fb_type_keys.keys():
            new_name = 'DFC-SSA ' + fb_type_keys[fb_type]
            for k in new_keys.keys():
                if k != "ndi_angles":
                    if os.path.exists(dir+"/"+k+"_"+fb_type+".csv"):
                        df = pd.read_csv(dir+"/"+k+"_"+fb_type+".csv", index_col=0)
                        toy_exp_df[new_keys[k]][new_name] = df.iloc[:,0]
                    elif os.path.exists(dir+"/"+k+"_"+fb_type+".npy"):
                        array = np.load(dir+"/"+k+"_"+fb_type+".npy")
                        toy_exp_df[new_keys[k]][new_name] = array
                    else:
                        print("Key "+k+" not found in "+dir)
            max_eig_vector = np.load(dir+"/"+max_eig_key+"_"+fb_type+".npy")
            toy_exp_df['eig_A_inst'][new_name] = \
                np.array([max_eig_vector[j] for j in range(2, len(max_eig_vector), 4)])
            toy_exp_df['eig_A'][new_name] = \
                np.array([max_eig_vector[j] for j in range(3, len(max_eig_vector), 4)])
        print('DFC-SSA processed')

    elif 'dfc_ss' in experiment.name.lower():
        max_eig_key = 'max_eig_bcn_mean'
        for fb_type in fb_type_keys.keys(): 
            new_name = 'DFC-SS ' + fb_type_keys[fb_type]
            for k in new_keys.keys():
                if os.path.exists(dir+"/"+k+"_"+fb_type+".csv"):
                    df = pd.read_csv(dir+"/"+k+"_"+fb_type+".csv", index_col=0)
                    toy_exp_df[new_keys[k]][new_name] = df.iloc[:, 0]

                elif os.path.exists(dir+"/"+k+"_"+fb_type+".npy"):
                    array = np.load(dir+"/"+k+"_"+fb_type+".npy")
                    toy_exp_df[new_keys[k]][new_name] = array
                else:
                    print("Key "+k+" not found in "+dir)
            max_eig_vector = np.load(dir+"/"+max_eig_key+"_"+fb_type+".npy")
            toy_exp_df['eig_A_inst'][new_name] = \
                np.array([max_eig_vector[j] for j in range(2, len(max_eig_vector), 4)])
            toy_exp_df['eig_A'][new_name] = \
                np.array([max_eig_vector[j] for j in range(3, len(max_eig_vector), 4)])
        print('DFC_SS processed')

    elif ('dfc' in experiment.name.lower()) and not ('nois' in experiment.name.lower()):
        max_eig_key = 'max_eig_bcn_mean'
        for fb_type in fb_type_keys.keys(): 
            new_name = 'DFC ' + fb_type_keys[fb_type]
            for k in new_keys.keys():
                if os.path.exists(dir+"/"+k+"_"+fb_type+".csv"):
                    df = pd.read_csv(dir+"/"+k+"_"+fb_type+".csv", index_col=0)
                    toy_exp_df[new_keys[k]][new_name] = df.iloc[:,0]

                elif os.path.exists(dir+"/"+k+"_"+fb_type+".npy"):
                    array = np.load(dir+"/"+k+"_"+fb_type+".npy")
                    toy_exp_df[new_keys[k]][new_name] = array
                else:
                    print("Key "+k+" not found in "+dir)
            max_eig_vector = np.load(dir+"/"+max_eig_key+"_"+fb_type+".npy")
            toy_exp_df['eig_A_inst'][new_name] = \
                np.array([max_eig_vector[j] for j in range(2, len(max_eig_vector), 4)])
            toy_exp_df['eig_A'][new_name] = \
                np.array([max_eig_vector[j] for j in range(3, len(max_eig_vector), 4)])
        print('DFC processed')

    elif 'dfa' in experiment.name.lower():
        toy_exp_df['gn_angles']['DFA'] = pd.read_csv(dir+"/gn_angles_network_DFA.csv", index_col=0)
        toy_exp_df['mn_angles']['DFA'] = pd.read_csv(dir+"/gnt_angles_network_DFA.csv", index_col=0)
        print('DFA processed')

    else:
        print('Experiment '+experiment.name+' not included.')

print('All processed')

ylabels = {
    'condition_1': 'Condition 1',
    'condition_2': 'Condition 2',
    'gn_angles': r'$\Delta W_i \hspace{0.5} \angle \hspace{0.5} \Delta W_i^{GN} \hspace{0.5} $ [$^\circ$]',
    'mn_angles': r'$\Delta W_i \hspace{0.5} \angle \hspace{0.5} \Delta W_i^{MN} \hspace{0.5} $ [$^\circ$]',
    'ssa_angles': r'$\Delta W_i \hspace{0.5} \angle \hspace{0.5} \Delta W_i^{SSA} \hspace{0.5} $ [$^\circ$]',
    'eig_A': r'$\lambda_{max}$',
}

def get_linestyle_color(experiment_name):
    """
    Decides the color based on the experiment type and the linestyle based on Train/FreezeFB.
    """
    if 'SSA' in experiment_name: color = "purple"
    elif 'SS' in experiment_name: color = "green"
    elif 'DFC' in experiment_name: color = "royalblue" 
    elif 'DFA' in experiment_name: color = "red" 
    else: raise "Experiment type "+experiment_name+" not known"

    if 'TrainFB' in experiment_name: linestyle = "-"
    elif 'FreezeFB' in experiment_name: linestyle = "--"
    elif 'DFA' in experiment_name: linestyle = "-"
    else: raise "Experiment type "+experiment_name+" not known"

    return linestyle, color

def plot_rolling(ax, data, label, linestyle, color, zorder, smooth=30, alpha_fill=0.1, line_width=1):
    """
    Plots a moving avg and var of the dat. Based on "plot_rolling" of figure_utils.
    """

    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame): data = data.to_numpy()
    df = pd.DataFrame(data, columns=["data"])
    df["roll_mean"] = df["data"].rolling(window=smooth, min_periods=1).mean()
    df["roll_std"] = df["data"].rolling(window=smooth, min_periods=1).std()
    df["roll_lower"] = df["roll_mean"].sub(df["roll_std"], fill_value=0)
    df["roll_upper"] = df["roll_mean"].add(df["roll_std"], fill_value=0)

    l, = ax.plot(np.arange(len(df["roll_mean"])), df["roll_mean"], label=label, linestyle=linestyle,
                 color=color, zorder=zorder, linewidth=line_width)
    ax.fill_between(np.arange(len(df["roll_mean"])), df["roll_lower"], df["roll_upper"], alpha=alpha_fill, lw=0,
                    color=color, zorder=zorder)
    return l

experiment_order = ['DFC TrainFB', 'DFC-SS TrainFB', 'DFC-SSA TrainFB', 'DFA',
                    'DFC FreezeFB', 'DFC-SS FreezeFB', 'DFC-SSA FreezeFB']
legend = {
    'DFC TrainFB': 'DFC',
    'DFC-SS TrainFB': 'DFC-SS',
    'DFC-SSA TrainFB': 'DFC-SSA',
    'DFC FreezeFB': 'DFC (fixed)',
    'DFC-SS FreezeFB': 'DFC-SS (fixed)',
    'DFC-SSA FreezeFB': 'DFC-SSA (fixed)',
    'DFA': 'DFA',
}

if linear:
    ylims = {
        'condition_1': (0, 1),
        'condition_2': (0.5, 1),
        'gn_angles': (0, 75),
        'mn_angles': (0, 75),
        'ssa_angles': (0, 20),
        'eig_A': (-0.7, -0.4),
    }
else:
    ylims = {
        'condition_1': (0, 0.45),
        'condition_2': (0.7, 1),
        'gn_angles': (0, 70),
        'mn_angles': (0, 70),
        'ssa_angles': (0, 8),
        'eig_A': (-0.6, -0.4), 
    }

zorders = {
    'DFC TrainFB': 10,
    'DFC-SS TrainFB': 8,
    'DFC-SSA TrainFB': 6,
    'DFC FreezeFB': 9,
    'DFC-SS FreezeFB': 7,
    'DFC-SSA FreezeFB': 5,
    'DFA': 4,
}

nrows = 2
ncolumns = 3
vsize = 3
hsize = 5
fontsize_ylabel = 16
fontsize_xlabel = 14
fontsize_legend = 15
ticklabelsize = 14
fig, axes = plt.subplots(nrows, ncolumns, sharex=True, sharey=False, figsize=(hsize*ncolumns, vsize*nrows))
axes = axes.reshape(-1,)
metrics = ['condition_2', 'condition_1', 'eig_A', 'mn_angles', 'gn_angles', 'ssa_angles']
line_list = []
for i, m in enumerate(metrics):
    for exp in experiment_order:
        if exp in toy_exp_df[m].keys():
            ls, color = get_linestyle_color(exp)
            if "SSA" in exp:
                line_width = 1
            else:
                line_width = 1

            l = plot_rolling(ax=axes[i], data=toy_exp_df[m][exp], label=legend[exp], linestyle=ls, color=color,
                             zorder=zorders[exp], line_width=line_width)
            if m == "mn_angles": line_list.append(l)
    axes[i].set_ylabel(ylabels[m], fontsize=fontsize_ylabel)
    axes[i].set_axis_on()
    axes[i].set_xlim(0,300)
    axes[i].xaxis.set_major_locator(ticker.FixedLocator([100,200,300]))
    if i < 2: axes[i].yaxis.set_major_locator(ticker.LinearLocator(numticks=4)) 
    if i == 2: axes[i].yaxis.set_major_locator(ticker.LinearLocator(numticks=5)) 
    if i >= 3: axes[i].yaxis.set_major_locator(ticker.MaxNLocator(4))
    axes[i].tick_params(labelsize=ticklabelsize)
    axes[i].set_ylim(ylims[m])
    sns.despine()
    if i>=3: axes[i].set_xlabel('Epoch', fontsize=fontsize_xlabel)

label_list = [legend[exp] for exp in experiment_order]
leg = axes[-1].legend(handles=line_list, labels=label_list, loc="upper right",
                      frameon=False, ncol=2, fontsize=fontsize_legend,
                      labelspacing=0.6, 
                      handlelength=1.5, 
                      handletextpad=0.5,
                      columnspacing=0.7,
                      markerscale=1.5,
                      borderpad=0.,
                      borderaxespad=0.,
                      )

plt.tight_layout()
plt.subplots_adjust(hspace=0.3)
name = 'fig3'
plt.savefig(os.path.join(out_dir, name+'.png'))