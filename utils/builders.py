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

from networks import tpdi_networks
from networks import bp_network
from target_propagation import direct_feedback_networks, \
    dtp_networks
import torch
import numpy as np
import random

def generate_data_from_teacher(args, num_train=1000, num_test=100, n_in=5, n_out=5,
                               n_hidden=[10,10,10], activation='tanh',
                               device=None, num_val=None):
    """
    Generate data for a regression task through a teacher model.
    This function generates random input patterns and creates a random MLP
    (fully-connected neural network), that is used as a teacher model. I.e., the
    generated input data is fed through the teacher model to produce target
    outputs. The so produced dataset can be used to train and assess a
    student model. Hence, a learning procedure can be verified by validating its
    capability of training a student network to mimic a given teacher network.
    Input samples will be uniformly drawn from a unit cube.
    .. warning::
        Since this is a synthetic dataset that uses random number generators,
        the generated dataset depends on externally configured random seeds
        (and in case of GPU computation, it also depends on whether CUDA
        operations are performed in a derterministic mode).
    Args:
        num_train (int): Number of training samples.
        num_test (int): Number of test samples.
        n_in (int): Passed as argument ``n_in`` to class
            :class:`networks.networks.DTPNetwork`
            when building the teacher model.
        n_out (int): Passed as argument ``n_out`` to class
            :class:`networks.networks.DTPNetwork`
            when building the teacher model.
        n_hidden (list): Passed as argument ``n_hidden`` to class
            :class:`networks.networks.DTPNetwork` when building the teacher model.
        activation (str): Passed as argument ``activation`` to
            class :class:`networks.networks.DTPNetwork` when building the
            teacher model
    Returns:
        See return values of function :func:`regression_cubic_poly`.
    """

    device = torch.device('cpu')
    if num_val is None:
        num_val = num_test
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    rand = np.random

    train_x = rand.uniform(low=-1, high=1, size=(num_train, n_in))
    test_x = rand.uniform(low=-1, high=1, size=(num_test, n_in))
    val_x = rand.uniform(low=-1, high=1, size=(num_val, n_in))

    teacher = dtp_networks.DTPNetwork(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                                      activation=activation, output_activation='linear',
                                      bias=True, initialization='teacher')

    if args.double_precision:
        train_y = teacher.forward(torch.from_numpy(train_x).to(torch.float64).to(device)) \
            .detach().cpu().numpy()
        test_y = teacher.forward(torch.from_numpy(test_x).to(torch.float64).to(device)) \
            .detach().cpu().numpy()
        val_y = teacher.forward(torch.from_numpy(val_x).to(torch.float64).to(device)) \
            .detach().cpu().numpy()
    else:
        train_y = teacher.forward(torch.from_numpy(train_x).float().to(device))\
            .detach().cpu().numpy()
        test_y = teacher.forward(torch.from_numpy(test_x).float().to(device))\
            .detach().cpu().numpy()
        val_y = teacher.forward(torch.from_numpy(val_x).float().to(device))\
            .detach().cpu().numpy()

    return train_x, test_x, val_x, train_y, test_y, val_y


def build_network(args):
    """
    Create the network based on the provided command line arguments
    Args:
        args: command line arguments
    Returns: a network
    """

    forward_requires_grad = args.forward_requires_grad
    if args.classification:
        assert args.output_activation == 'softmax', "Output layer should " \
                    "represent probabilities => use softmax"
        output_activation = 'linear'
        
    else:
        output_activation = args.output_activation

    kwargs_bp = {
        'n_in': args.size_input,
        'n_hidden': args.size_hidden,
        'n_out': args.size_output,
        'activation': args.hidden_activation,
        'bias': not args.no_bias,
        'initialization': args.initialization,
        'output_activation': output_activation,
    }

    kwargs_dtp = {
              'sigma': args.sigma,
              'forward_requires_grad': forward_requires_grad,
              'save_df': args.save_df,
              'fb_activation': args.fb_activation,
              }

    kwargs_dfc = {'ndi': args.ndi,
                  'alpha_di': args.alpha_di,
                  'dt_di': args.dt_di,
                  'dt_di_fb': args.dt_di_fb,
                  'tmax_di': args.tmax_di,
                  'tmax_di_fb': args.tmax_di_fb,
                  'epsilon_di': args.epsilon_di,
                  'reset_K': args.reset_K,
                  'initialization_K': args.initialization_K,
                  'noise_K': args.noise_K,
                  'compare_with_ndi': args.compare_with_ndi,
                  'out_dir': args.out_dir,
                  'learning_rule': args.learning_rule,
                  'use_initial_activations': args.use_initial_activations,
                  'homeostatic_constant': args.c_homeostatic,
                  'sigma': args.sigma,
                  'sigma_fb': args.sigma_fb,
                  'sigma_output_fb': args.sigma_output_fb,
                  'forward_requires_grad': forward_requires_grad,
                  'save_df': args.save_df,
                  'clip_grad_norm': args.clip_grad_norm,
                  'k_p': args.k_p,
                  'k_p_fb': args.k_p_fb,
                  'inst_system_dynamics': args.inst_system_dynamics,
                  'alpha_fb': args.alpha_fb,
                  'noisy_dynamics': args.noisy_dynamics,
                  'fb_learning_rule': args.fb_learning_rule,
                  'inst_transmission': args.inst_transmission,
                  'inst_transmission_fb': args.inst_transmission_fb,
                  'time_constant_ratio': args.time_constant_ratio,
                  'time_constant_ratio_fb': args.time_constant_ratio_fb,
                  'apical_time_constant': args.apical_time_constant,
                  'apical_time_constant_fb': args.apical_time_constant_fb,
                  'grad_deltav_cont': args.grad_deltav_cont,
                  'homeostatic_wd_fb': args.homeostatic_wd_fb,
                  'efficient_controller': args.efficient_controller,
                  'proactive_controller': args.proactive_controller,
                  'save_NDI_updates': args.save_NDI_angle,
                  'save_eigenvalues': args.save_eigenvalues,
                  'save_eigenvalues_bcn': args.save_eigenvalues_bcn,
                  'save_norm_r': args.save_norm_r,
                  'simulate_layerwise': args.simulate_layerwise,
                  'include_non_converged_samples': args.include_non_converged_samples,
                  }

    kwargs_dfc = {**kwargs_bp, **kwargs_dfc}
    kwargs_dtp = {**kwargs_bp, **kwargs_dtp}

    if args.network_type == 'DTP':
        net = dtp_networks.DTPNetwork(**kwargs_dtp)

    elif args.network_type == 'BP':
        net = bp_network.BPNetwork(**kwargs_bp)
    elif args.network_type == 'DFC':
        net = tpdi_networks.DFCNetwork(**kwargs_dfc)
    elif args.network_type == 'DFA':
        net = direct_feedback_networks.DDTPMLPNetwork(**kwargs_dtp,
                              size_hidden_fb=None,
                              fb_hidden_activation='linear',
                              recurrent_input=False,
                              fb_weight_initialization=args.initialization_K)
    else:
        raise ValueError('The provided network type {} is not supported'.format(
            args.network_type
        ))

    return net