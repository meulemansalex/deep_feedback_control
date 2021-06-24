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
This is a script for creating figure 1D.
"""


import importlib
import os
import sys
import argparse
import random
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from argparse import Namespace
from torch.utils.data import DataLoader
from utils.train import train_only_feedback_parameters
from utils.args import parse_cmd_arguments
from utils import builders, utils
from tensorboardX import SummaryWriter
import os.path
import pickle


def _override_cmd_arg(config):
    sys.argv = [sys.argv[0]]
    for k, v in config.items():
        if isinstance(v, bool):
            cmd = '--%s' % k if v else ''
        else:
            cmd = '--%s=%s' % (k, str(v))
        if not cmd == '':
            sys.argv.append(cmd)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_module', type=str,
                        default='configs.toy_experiments.config_fig1D',
                        help='The name of the module containing the config.')
    parser.add_argument('--out_dir', type=str,
                        default='./logs/toy_experiments/fig1D',
                        help='Directory where the trained FB network is to be found/saved'
                             'and the results will be saved.')
    args = parser.parse_args()
    config_module = importlib.import_module(args.config_module)
    _override_cmd_arg(config_module.config)
    out_dir = args.out_dir
    args = parse_cmd_arguments()
    args.out_dir = out_dir

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    if args.cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print('Using cuda: ' + str(use_cuda))

    if args.double_precision:
        torch.set_default_dtype(torch.float64)

    if args.dataset == 'student_teacher':
        print('### Training a student_teacher regression model ###')
        torch.set_default_tensor_type('torch.FloatTensor')
        if args.double_precision:
            torch.set_default_dtype(torch.float64)

        if not args.load_ST_dataset:
            train_x, test_x, val_x, train_y, test_y, val_y = \
                builders.generate_data_from_teacher(
                    n_in=args.size_input, n_out=args.size_output,
                    n_hidden=[1000, 1000, 1000, 1000], device=device,
                    num_train=args.num_train, num_test=args.num_test,
                    num_val=args.num_val,
                    args=args, activation='relu')
        else:
            train_x = np.load('./data/train_x.npy')
            test_x = np.load('./data/test_x.npy')
            val_x = np.load('./data/val_x.npy')
            train_y = np.load('./data/train_y.npy')
            test_y = np.load('./data/test_y.npy')
            val_y = np.load('./data/val_y.npy')

        if use_cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if args.double_precision:
            torch.set_default_dtype(torch.float64)
        
        train_loader = DataLoader(utils.RegressionDataset(train_x, train_y,
                                                          args.double_precision),
                                  batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(utils.RegressionDataset(test_x, test_y,
                                                         args.double_precision),
                                 batch_size=args.batch_size, shuffle=False)

        if args.no_val_set:
            val_loader = None
        else:
            val_loader = DataLoader(utils.RegressionDataset(val_x, val_y,
                                                            args.double_precision),
                                    batch_size=args.batch_size, shuffle=False)

    else:
        raise ValueError('The provided dataset {} is not supported.'.format(
            args.dataset
        ))

    if args.log_interval is None:
        args.log_interval = max(1, int(len(train_loader)/100))

    if args.save_logs:
        writer = SummaryWriter(logdir=args.out_dir)
    else:
        writer = None

    summary = utils.setup_summary_dict(args)
    net = builders.build_network(args)
    net.to(device)

    net.train()
    net.zero_grad()
    train_var = Namespace()
    train_var.summary = summary
    train_var.forward_optimizer, train_var.feedback_optimizer, \
    train_var.feedback_init_optimizer = utils.choose_optimizer(args, net)
    train_var.loss_function = nn.MSELoss()
    if args.compute_gn_condition_init and args.epochs_fb > 0:
        train_var.gn_condition_init = np.array([])

    savepath = os.path.join(args.out_dir,"trainedFB.pickle")
    if os.path.exists(savepath):
        print("Loading already existing pretrained FB.")
        print("If you want to re-train the FB, please delete the file "+savepath)

        with open(savepath, "rb") as f:
            feedback_parameters = pickle.load(f)

        for i, layer in enumerate(net.layers):
            layer.feedbackweights = feedback_parameters[i]

        compute_gn_condition = True
        train_var.gn_condition_init = np.array([])
        for i, (inputs, targets) in enumerate(train_loader):
            predictions = net.forward(inputs)
            if compute_gn_condition and i % args.log_interval == 0:
                gn_condition = net.compute_condition_two(retain_graph=False)
                train_var.gn_condition_init = np.append(train_var.gn_condition_init,
                                                        gn_condition.item())
        train_var.summary['gn_condition_init'] = np.mean(train_var.gn_condition_init)
        print('GN condition init: ', train_var.summary['gn_condition_init'])

    else:
        print("Feedback weights being trained.")

        train_var.epochs_init = 0
        train_var.batch_idx = 1
        train_var.batch_idx_fb = 1
        train_var.init_idx = 1
        train_var.gn_condition_init = np.array([])

        for e_fb in range(args.epochs_fb):
            print('Feedback weights training: epoch {}'.format(e_fb))
            train_var.epochs_init = e_fb
            if e_fb == args.epochs_fb - 1:
                compute_gn_condition = True
            else:
                compute_gn_condition = False

            train_only_feedback_parameters(args, train_var, device, train_loader,
                                           net, writer,
                                           compute_gn_condition=compute_gn_condition,
                                           init=True)

        print(f'Feedback weights initialization done after {args.epochs_fb} epochs.')
        train_var.summary['gn_condition_init'] = np.mean(train_var.gn_condition_init)
        feedback_parameters = net.get_feedback_parameter_list()
        with open(savepath, "wb") as f:
            pickle.dump(feedback_parameters, f)
        print('GN condition init: ', train_var.summary['gn_condition_init'])

    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        predictions = net.forward(inputs)
        train_var.forward_optimizer.zero_grad()
        loss = train_var.loss_function(predictions, targets)
        target_lr = args.target_stepsize
        output_target = net.compute_output_target(loss, target_lr)
        v_feedforward = [l.linearactivations for l in net.layers]
        r_feedforward = [l.activations for l in net.layers]

        tmax = np.round(net.tmax_di).astype(int)
        batch_size = net.layers[0].activations.shape[0]

        r_target_pi, u_pi, _, _ = \
            net.controller(output_target, net.alpha_di, net.dt_di, tmax,
                            mode=net.simulation_mode,
                            inst_system_dynamics=False,
                            k_p=net.k_p,
                            noisy_dynamics=False,
                            inst_transmission=net.inst_transmission,
                            time_constant_ratio=net.time_constant_ratio,
                            apical_time_constant=net.apical_time_constant,
                            sparse=True,
                            proactive_controller=net.proactive_controller,
                            sigma=net.sigma)

        n_samples = 3
        for sample in range(n_samples):
            plt.figure()
            plt.plot(r_target_pi[-1][:,sample,:].detach().cpu(),'b',label="Network output")
            plt.hlines(output_target[sample].detach().cpu(), 0, tmax, 'k', "dashed", label="Output target")
            plt.title("Proportional + Integral control")
            plt.xlabel("Timesteps")
            plt.ylabel("Output value")
            plt.savefig(os.path.join(args.out_dir,"PI_control"+str(sample)+".png"))
            plt.close()

            fig,ax = plt.subplots()
            ax.plot(r_target_pi[-1][1:, sample, :].detach().cpu(), 'b',
                     label="Network output")
            ax.hlines(output_target[sample].detach().cpu(), 0, tmax, 'k',
                       "dashed", label="Output target")
            ax2=ax.twinx()
            ax2.plot(u_pi[1:, sample, :].detach().cpu(), 'r',
                     label="Control input")
            plt.title("Proportional + Integral control")
            plt.savefig(os.path.join(args.out_dir, "PI_control_signal" + str(sample) + ".png"))
            plt.close()

        return summary

if __name__ == '__main__':
    run()