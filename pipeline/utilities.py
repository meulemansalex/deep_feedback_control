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

import json
import os
import numpy as np
import pandas as pd


def save_list(lst, filename):
    with open(filename, 'w') as f:
        for item in lst:
            f.write('%s\n' % item)


def save_dict_to_json(dictionary, file_name):
    with open(file_name, 'w') as json_file:
        json.dump(dictionary, json_file)


def read_list(filename):
    lst = []
    with open(filename, 'r') as f:
        for line in f:
            lst.append(line[:-1])
    return lst


def save_result_dict(result_dict, out_dir):
    keys = []
    names = []
    is_array = np.array([])

    for key, sub_dict in result_dict.items():

        if len(sub_dict)>0:
            keys.append(key)
            np.append(is_array, isinstance(sub_dict[list(sub_dict.keys())[0]],
                                           np.ndarray))
            for name, value in sub_dict.items():
                file_name = os.path.join(out_dir, key + '_' + name)
                if len(names) < len(sub_dict):
                    names.append(name)
                if isinstance(value, np.ndarray):
                    np.save(file_name + '.npy', value)
                if isinstance(value, pd.DataFrame):
                    value.to_csv(file_name + '.csv')

    save_list(keys, os.path.join(out_dir, 'keys.txt'))
    save_list(names, os.path.join(out_dir, 'names.txt'))
    np.save(os.path.join(out_dir, 'is_array.npy'), is_array)


def read_result_dict(result_dir):
    keys = read_list(os.path.join(result_dir, 'keys.txt'))
    names = read_list(os.path.join(result_dir, 'names.txt'))
    is_array = np.load(os.path.join(result_dir, 'is_array.npy'))

    result_dict = {}

    for i, key in enumerate(keys):
        result_dict[key] = {}
        for name in names:
            file_name = os.path.join(result_dir, key + '_' + name)
            if not 'angle' in file_name:
                if os.path.exists(file_name + '.npy'):
                    result_dict[key][name] = np.load(file_name + '.npy')
            else:
                if os.path.exists(file_name + '.csv'):
                    result_dict[key][name] = pd.read_csv(file_name + '.csv',
                                                         index_col=0)

    return result_dict


def postprocess_summary(summary, name, result_dict, result_keys):
    """
    Save the result_keys performances in the result_dict.
    """

    for key in result_keys:
        if key in summary.keys():
            result_dict[key][name] = summary[key]


def create_layerwise_result_dict(result_key_dict):
    """
    Sort the result_key_dict (result dict of only one performance key)
    with layer numbers as keys and as values Dataframes with the training method
    names as column names.
    """
    
    layerwise_dict = {}
    nb_layers = len(result_key_dict[list(result_key_dict.keys())[0]].columns)
    for i in range(nb_layers):
        layerwise_dict[i] = pd.DataFrame()

    column_names = []
    for name, dtframe in result_key_dict.items():
        column_names.append(name)
        for i in range(nb_layers):
            layerwise_dict[i] = pd.concat([layerwise_dict[i], dtframe[i]],
                                          axis=1)
    return layerwise_dict, column_names