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
import itertools

def run(config_file_arg=None):
    """run the code to generate the config file for the pipeline

    Parameters
    ----------
    config_file_name : str
        path (including the name of the config file) where the file is generated.
    Return
    -------
    None
    """

    config_params = {
        'random_seed': [42, 7, 13],
    }

    additional_params = {
        'num_hidden': 2,
        'size_input': 15,
        'size_hidden': 10,
        'size_output': 5,
        'hidden_activation': 'linear',
        'output_activation': 'linear',
        'epochs': 5,
        'batch_size': 128,
        'optimizer': 'SGD',
        'optimizer_fb': 'SGD',
        'momentum': 0.0,
        'no_cuda': False,
        'epochs_fb': 0.,
        'extra_fb_epochs': 2.0,
        'initialization_K': 'weight_product',
        'fb_learning_rule': 'normal_controller',
        'freeze_fb_weights': True,
        'save_logs': True,
        'save_eigenvalues': True,
        'save_eigenvalues_bcn': True,
        'save_norm_r': True,
        'save_BP_angle': False,
        'save_GN_angle': True,
        'save_GNT_angle': True,
        'save_NDI_angle': False,
        'compare_with_ndi': False,
        'save_condition_gn': True,
        'save_df': False,
        'log_interval': 100,
    }
    
    if not config_file_arg:
        config_file_name = 'config_example'

    if not config_file_name[:7] == 'config_':
        config_file_name = 'config_' + config_file_name

    fixed_params = {
        # dataset_args
        'dataset': 'student_teacher',
        'num_train': 1000,
        'num_test': 1000,
        'num_val': 1000,
        'no_preprocessing_mnist': False,
        'no_val_set': False,

        # training_args
        'epochs': 100,
        'batch_size': 128,
        'lr': 0.1,
        'lr_fb': 0.1,
        'target_stepsize': 0.001,
        'optimizer': 'Adam',
        'optimizer_fb': 'Adam',
        'momentum': 0.0,
        'sigma': 0.08,
        'forward_wd': 0,
        'feedback_wd': 0,
        'train_parallel': False,  
        'normalize_lr': True,   
        'epochs_fb': 0,
        'freeze_forward_weights': False,
        'freeze_fb_weights': False,
        'shallow_training': False,
        'extra_fb_epochs': 0,
        'extra_fb_minibatches': 0,
        'only_train_first_layer': False,
        'train_only_feedback_parameters': False,
        'clip_grad_norm': -1,

        # adam_args
        'beta1': 0.99,
        'beta2': 0.99,
        'epsilon': 1e-8,
        'beta1_fb': 0.99,
        'beta2_fb': 0.99,
        'epsilon_fb': 1e-8,

        # network_args
        'num_hidden': 5,
        'size_hidden': 5,
        'size_input': 10,
        'size_output': 2,
        'hidden_activation': 'tanh',
        'output_activation': 'linear',
        'no_bias': False,
        'network_type': 'DFC',
        'initialization': 'xavier_normal',
        'fb_activation': 'tanh',

        # miscellaneous_args
        'no_cuda': False,
        'random_seed': 42,
        'cuda_deterministic': False,
        'hpsearch': False,
        'multiple_hpsearch': False,
        'single_precision': False,
        'evaluate': False,

        # logging_args
        'out_dir': None,
        'save_logs': True,
        'save_BP_angle': True,
        'save_GN_angle': True,
        'save_GNT_angle': True,
        'save_NDI_angle': True,
        'save_condition_gn': True,
        'save_df': True,
        'gn_damping': 0.0,
        'log_interval': 10,
        'gn_damping_hpsearch': False,
        'save_nullspace_norm_ratio': True,
        'save_fb_statistics_init': True,
        'make_dynamics_plot': False,

        # dynamical_inversion_args
        'ndi': False,
        'alpha_di': 0.001,
        'dt_di': 0.1,
        'tmax_di': 300.,
        'epsilon_di': 0.5,
        'reset_K': False,
        'initialization_K': 'xavier_normal',
        'noise_K': 0.,
        'compare_with_ndi': False,

        # dfc_args
        'learning_rule': 'voltage_difference',
        'use_initial_activations': False,
        'beta_homeostatic': 0.01,
        'c_homeostatic': -1,
        'k_p': 1.,
        'inst_system_dynamics': False,
        'alpha_fb': 0.5,
        'noisy_dynamics': False,
        'fb_learning_rule': 'normal_controller',
        'time_constant_ratio': 1.,
        'apical_time_constant': 0.1,
        'inst_transmission': False,
        'grad_deltav_cont': True,
        'apical_time_constant': 0.5,
        }

    # overwrite defaults in fixed_params with those manually set in additional params
    fixed_params.update(additional_params)

    # delete those entries of fixed_params that are in config_params
    for key in config_params.keys():
        if key in fixed_params:
            del fixed_params[key]

    result_keys = [
        'loss_test',
        'loss_train',
        'bp_angles',
        'gnt_angles',
        'gn_angles',
        'ndi_angles',
        'condition_gn',
        'nullspace_relative_norm_angles',
        'converged',
        'not_converged',
        'diverged',
        ]

    # statistics already computed during the run
    # angle statistics are later computed and added to results.csv
    statistics_keys = [
        'loss_test_val_best',
        'loss_test_best',
        'loss_train_best',
        'loss_test_last',
        'loss_train_last',
        'loss_val_last',
        'loss_val_best',
        ]

    if fixed_params['dataset'] in ['mnist', 'fashion_mnist', 'cifar10']:
        # in case of a classification dataset, add accuracy statistics
        statistics_keys += [
            'acc_test_val_best',
            'acc_test_best',
            'acc_train_best',
            'acc_test_last',
            'acc_train_last',
            'acc_val_last',
            'acc_val_best',
            ]

    cwd = os.getcwd()
    if config_file_arg:
        #TODO: remove files from folder if it exists
        directory = os.path.dirname(config_file_arg)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        config_file_arg += '.py'
        config_file_path = config_file_arg
        config_file_name = os.path.basename(config_file_arg)
    else:
        config_file_path = os.path.join(cwd, config_file_name)
        if not config_file_path[-3:] == '.py': config_file_path+='.py'

    with open(config_file_path, 'w') as f:
        keys, values = zip(*config_params.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        list_names = []

        # write all possible combinations
        for p in permutations_dicts:
            # use the values as name of the run
            name = ''
            separator = '__'
            for key, value in p.items():
                name += str(value).replace('.','_') + separator
            name = name[:-len(separator)]  # remove final underscore
            list_names.append(name)  # save all names for config_collection list
            write_dict_to_txt(f, p, 'config_'+name)

        # write config_collection
        f.write('config_collection = {\n')
        for name in list_names:
            f.write("'" + fixed_params['network_type'] + '_' + name + "': config_" + name + ",\n")
        f.write('}\n\n')

        # write fixed parameters, result keys, etc...
        write_array_to_txt(f, result_keys, 'result_keys')
        write_array_to_txt(f, statistics_keys, 'statistics_keys')
        write_dict_to_txt(f, fixed_params, 'config_fixed')

        print('Config file '+config_file_name+' successfully generated.')


def write_dict_to_txt(file_handler, dct, name):
    """Write the dict to a txt file in python format (so the python code
    for creating a dictionary with the same content as dct).

    Args:
        file_handler: python file handler of the file where the dct will be
            written to.
        dct: dictionary
        name: name of the dictionary
    """
    file_handler.write(name + ' = {\n')
    for key, value in dct.items():
        if isinstance(value, str):
            value = "'" + value + "'"
        else:
            value = str(value)
        file_handler.write("'" + key + "': " + value + ",\n")
    file_handler.write('}\n\n')


def write_array_to_txt(file_handler, arr, name):
    """Write the dict to a txt file in python format (so the python code
    for creating a dictionary with the same content as dct).

    Args:
        file_handler: python file handler of the file where the dct will be
            written to.
        dct: dictionary
        name: name of the dictionary
    """
    file_handler.write(name + ' = [\n')
    for value in arr:
        if isinstance(value, str):
            value = "'" + value + "'"
        else:
            value = str(value)
        file_handler.write(value + ",\n")
    file_handler.write(']\n\n')

if __name__ == '__main__':
    run()
