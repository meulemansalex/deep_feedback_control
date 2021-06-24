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
This is a script to run a model for several hyperparameter 
fixed configurations.
"""


import numpy as np
import importlib
import main
import argparse
import os
import sys
import pandas as pd
import pickle
from datetime import datetime
from pipeline import figure_utils
from pipeline.utilities import *

def run(args=None, config_file_arg=None):

    if not args:
        parser = argparse.ArgumentParser()
        parser.add_argument('--out_dir', type=str, default='logs/pipeline',
                            help='Directory where the results will be saved.')
        parser.add_argument('--experiment_name', type=str, default='experiment',
                            help='Subdirectory inside outdir where the results will be saved.')
        parser.add_argument('--config_module', type=str,
                            default='configs.configs_mnist.mnist_ndi',
                            help='The name of the module containing the configs.')
        args = parser.parse_args()
    
    now = datetime.now()
    if config_file_arg:
        args.out_dir = config_file_arg.config_file_path + '__' + now.strftime("%Y_%m_%d__%H_%M_%S")
    else:
        args.out_dir += '/' + args.experiment_name + '__' + now.strftime("%Y_%m_%d__%H_%M_%S")

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if config_file_arg:
        config_module = config_file_arg.config_file_obj
    else:
        config_module = importlib.import_module(args.config_module)

    config_collection = config_module.config_collection
    config_fixed = config_module.config_fixed
    result_keys = config_module.result_keys
    if 'DFA' in config_collection.keys():
        result_keys_dfa = config_module.result_keys_DFA

    result_dict = {}
    for key in result_keys:
        result_dict[key] = {}

    result_dict_path = os.path.join(args.out_dir, 'result_dict')
    if not os.path.exists(result_dict_path):
        os.makedirs(result_dict_path)

    columns = config_module.statistics_keys
    index = [name for name, config in config_collection.items()]
    results = pd.DataFrame(index=index, columns=columns)

    for name, config in config_collection.items():
        config['out_dir'] = os.path.join(args.out_dir, name)
        summary = run_training(config, config_fixed)
        for key in columns:
            if key in summary.keys():
                results.loc[name, key] = summary[key]

        if config_fixed['save_df']:
            if name != 'DFA':
                results = add_avg_angles(results, summary, name, result_keys)
            else:
                results = add_avg_angles(results, summary, name, result_keys_dfa)

        results.to_csv(os.path.join(args.out_dir, 'results.csv'))

        postprocess_summary(summary, name, result_dict, result_keys)
        
        filename = os.path.join(args.out_dir, 'result_dict.pickle')
        with open(filename, 'wb') as f:
            pickle.dump(result_dict, f)
        save_result_dict(result_dict, result_dict_path)

        print('\n\t----------------------')
        print('Experiment '+name+' finished!')
        print('\n\t----------------------')

    if "fig3" in args.config_module and "dfa" not in args.config_module:
        from pipeline import figure_S1_utils
        figure_S1_utils.run(args)
    else:
        figure_utils.plot_pipeline(args, result_keys, result_dict, config_fixed)

    print('Pipeline finished!')


def add_avg_angles(results, summary, name, result_keys):

    for key in result_keys:
        if 'angles' in key and not summary[key].empty:
            mean_per_layer = summary[key].mean(0)
            results.loc[name, 'avg_' + key] = np.mean(mean_per_layer)

    for key in result_keys:
        if 'angles' in key and not summary[key].empty:
            angles = summary[key]
            L = len(angles)
            m = L // 2  # middle point
            results.loc[name, key + 'avgfirsthalf'] = np.mean(np.mean(angles.iloc[:m, :]))
            results.loc[name, key + 'avgsecondhalf'] = np.mean(np.mean(angles.iloc[m:, :]))

    for key in result_keys:
        if 'angles' in key and not summary[key].empty:
            mean_per_layer = summary[key].mean(0)
            for i, m in enumerate(mean_per_layer):
                results.loc[name, 'avg_'+key+'_layer'+str(i)] = m

    for key in result_keys:
        if 'angles' in key and not summary[key].empty:
            first_angles = summary[key].iloc[0,:]
            for i, f in enumerate(first_angles):
                results.loc[name, 'initial_'+key+'_layer'+str(i)] = f

    for key in result_keys:
        if 'angles' in key and not summary[key].empty:
            angles = summary[key]
            L = len(angles)
            m = L // 2
            for layer in range(angles.shape[1]):
                results.loc[name, key+'avgfirsthalf'+'_layer'+str(layer)] = np.mean(angles.iloc[:m,layer])
                results.loc[name, key+'avgsecondhalf'+'_layer'+str(layer)] = np.mean(angles.iloc[m:,layer])

    return results


def _override_cmd_arg(config, fixed_space):
    sys.argv = [sys.argv[0]]
    for key in config.keys():
        if key in fixed_space:
            del fixed_space[key]
    for k, v in fixed_space.items():
        if isinstance(v, bool):
            cmd = '--%s' % k if v else ''
        else:
            cmd = '--%s=%s' % (k, str(v))
        if not cmd == '':
            sys.argv.append(cmd)
    for k, v in config.items():
        if isinstance(v, bool):
            cmd = '--%s' % k if v else ''
        else:
            cmd = '--%s=%s' % (k, str(v))
        if not cmd == '':
            sys.argv.append(cmd)


def run_training(config, config_fixed={}):
    """ Run the main file with the given config file and save the results of the
    alignment angles in the angle_dict"""

    _override_cmd_arg(config, config_fixed)
    summary = main.run()

    return summary


if __name__ == '__main__':
    run()