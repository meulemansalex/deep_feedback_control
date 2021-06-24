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
Collection of train and test functions.
"""

import os
from argparse import Namespace
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tensorboardX import SummaryWriter
import pandas as pd
from torchvision.utils import save_image
from utils import utils
import pickle
import time


def train(args, device, train_loader, net, writer, test_loader, summary,
          val_loader):
    """
    Train the given network on the given training dataset with DTP.
    Args:
        args (Namespace): The command-line arguments.
        device: The PyTorch device to be used
        train_loader (torch.utils.data.DataLoader): The data handler for
            training data
        net (DTPNetwork): The neural network
        writer (SummaryWriter): TensorboardX summary writer to save logs
        test_loader (DataLoader): The data handler for the test data
        summary (dict): summary dictionary with the performance measures of the
            training and testing
        val_loader (torch.utils.data.DataLoader): The data handler for the
            validation data
    """

    print('Training network ...')
    net.train()
    net.zero_grad()

    train_var = Namespace()
    train_var.summary = summary
    train_var.forward_optimizer, train_var.feedback_optimizer, \
    train_var.feedback_init_optimizer = utils.choose_optimizer(args, net)
    if args.classification:
        if args.output_activation == 'softmax':
            train_var.loss_function = nn.CrossEntropyLoss()
        else:
            raise ValueError('The mnist dataset can only be combined with a '
                             'softmax output activation.')

    elif args.regression or args.autoencoder:
        train_var.loss_function = nn.MSELoss()
    else:
        raise ValueError('The provided dataset {} is not supported.'.format(
            args.dataset
        ))
    train_var.batch_idx = 1
    train_var.batch_idx_fb = 1
    train_var.init_idx = 1

    train_var.epoch_losses = np.array([])
    train_var.epoch_times = np.array([])
    train_var.test_losses = np.array([])
    train_var.val_losses = np.array([])

    train_var.val_loss = None
    train_var.val_accuracy = None

    if args.compute_gn_condition_init and args.epochs_fb > 0:
        train_var.gn_condition_init = np.array([])
    else:
        train_var.gn_condition_init = -1

    if args.classification:
        train_var.epoch_accuracies = np.array([])
        train_var.test_accuracies = np.array([])
        train_var.val_accuracies = np.array([])
    if args.save_convergence:
        train_var.converged = np.array([])
        train_var.not_converged = np.array([])
        train_var.diverged = np.array([])

    if args.compare_with_ndi:
        train_var.rel_dist_to_NDI_vector = np.array([])
    if args.save_eigenvalues:
        train_var.max_eig_mean_vector = np.array([])
        train_var.max_eig_std_vector = np.array([])
    if args.save_eigenvalues_bcn:
        train_var.max_eig_bcn_mean_vector = np.array([])
        train_var.max_eig_bcn_std_vector = np.array([])
    if args.save_norm_r:
        train_var.norm_r_mean_vector = np.array([])
        train_var.norm_r_std_vector = np.array([])
        train_var.dev_r_mean_vector = np.array([])
        train_var.dev_r_std_vector = np.array([])

    if args.epochs_fb == 0 or args.freeze_fb_weights:
        print("Feedback weights not being trained.")
    else:
        print("Feedback weights being trained.")

        train_var.epochs_init = 0
        for e_fb in range(args.epochs_fb):
            print('Feedback weights training: epoch {}'.format(e_fb))
            train_var.epochs_init = e_fb
            if args.compute_gn_condition_init and e_fb == args.epochs_fb-1:
                compute_gn_condition = True
            else:
                compute_gn_condition = False

            train_only_feedback_parameters(args, train_var, device, train_loader,
                                           net, writer,
                                           compute_gn_condition=compute_gn_condition,
                                           init=True)
            if net.contains_nans():
                print('Network contains NaNs, terminating training.')
                break

        print(f'Feedback weights initialization done after {args.epochs_fb} epochs.')

        train_var.summary['gn_condition_init'] = np.mean(train_var.gn_condition_init)

        if args.train_only_feedback_parameters:
            print('Terminating training.')
            return train_var.summary

    train_var.epochs = 0

    params_epochs = [[p.data] for p in net.get_forward_parameter_list()]
    for e in range(args.epochs):
        if args.reset_K and e > 0:
            net.feedbackweights_initialization()
            print('Reseting forward optimizer')

        if args.save_convergence:
            net.converged_samples_per_epoch = 0
            net.diverged_samples_per_epoch = 0
            net.not_converged_samples_per_epoch = 0
            net.epoch = e

            if args.make_dynamics_plot and e % args.make_dynamics_plot_interval == 0:
                net.makeplots = True
            else:
                net.makeplots = False

        train_var.epochs = e
        if args.classification:
            train_var.accuracies = np.array([])
        train_var.losses = np.array([])

        if args.compare_with_ndi:
            net.rel_dist_to_NDI = []
        else:
            net.rel_dist_to_NDI = None

        epoch_initial_time = time.time()
        train_separate(args, train_var, device, train_loader, net, writer)
        if not args.freeze_fb_weights:
            for extra_e in range(args.extra_fb_epochs):
                train_only_feedback_parameters(args, train_var, device,
                                                   train_loader,
                                                   net, writer, log=False)

        if not args.no_val_set:
            train_var.val_accuracy, train_var.val_loss = \
                test(args, device, net, val_loader,
                         train_var.loss_function, train_var)

        train_var.test_accuracy, train_var.test_loss = \
            test(args, device, net, test_loader,
                     train_var.loss_function, train_var)

        train_var.epoch_time = time.time() - epoch_initial_time

        train_var.epoch_loss = np.mean(train_var.losses)

        print('\nEpoch {} '.format(e + 1))
        print('\ttraining loss = {}'.format(np.round(train_var.epoch_loss, 6)))
        if not args.no_val_set:
            print('\tval loss = {}'.format(np.round(train_var.val_loss, 6)))
        print('\ttest loss = {}'.format(np.round(train_var.test_loss, 6)))

        if args.classification:
            train_var.epoch_accuracy = np.mean(train_var.accuracies)
            print('\ttraining acc  = {} %'.format(np.round(train_var.epoch_accuracy * 100, 6)))
            if not args.no_val_set:
                print('\tval acc  = {} %'.format(np.round(train_var.val_accuracy * 100, 6)))
            print('\ttest acc  = {} %'.format(np.round(train_var.test_accuracy * 100, 6)))
        else:
            train_var.epoch_accuracy = None

        if args.compare_with_ndi:
            print('\tMean distance to NDI  = {}'.format(np.round(np.mean(net.rel_dist_to_NDI), 18)))
        print('\tepoch time = {} seconds'.format(np.round(train_var.epoch_time, 1)))

        if args.save_logs:
            if args.save_convergence: 

                avg_dist_to_NDI = None if not args.compare_with_ndi else np.mean(net.rel_dist_to_NDI)

                utils.save_logs_convergence(writer, step=e + 1, net=net,
                                            loss=train_var.epoch_loss,
                                            epoch_time=train_var.epoch_time,
                                            accuracy=train_var.epoch_accuracy,
                                            test_loss=train_var.test_loss,
                                            val_loss=train_var.val_loss,
                                            test_accuracy=train_var.test_accuracy,
                                            val_accuracy=train_var.val_accuracy,
                                            converged_samples_per_epoch=net.converged_samples_per_epoch,
                                            diverged_samples_per_epoch=net.diverged_samples_per_epoch,
                                            not_converged_samples_per_epoch=net.not_converged_samples_per_epoch,
                                            dist_to_NDI=avg_dist_to_NDI)
            else:
                utils.save_logs(writer, step=e + 1, net=net,
                                loss=train_var.epoch_loss,
                                epoch_time=train_var.epoch_time,
                                accuracy=train_var.epoch_accuracy,
                                test_loss=train_var.test_loss,
                                val_loss=train_var.val_loss,
                                test_accuracy=train_var.test_accuracy,
                                val_accuracy=train_var.val_accuracy)

        train_var.epoch_losses = np.append(train_var.epoch_losses,
                                           train_var.epoch_loss)
        train_var.test_losses = np.append(train_var.test_losses,
                                          train_var.test_loss)
        train_var.epoch_times = np.append(train_var.epoch_times,
                                          train_var.epoch_time)

        if not args.no_val_set:
            train_var.val_losses = np.append(train_var.val_losses,
                                             train_var.val_loss)

        if args.classification:
            train_var.epoch_accuracies = np.append(train_var.epoch_accuracies,
                                                   train_var.epoch_accuracy)
            train_var.test_accuracies = np.append(train_var.test_accuracies,
                                                  train_var.test_accuracy)
            if not args.no_val_set:
                train_var.val_accuracies = np.append(train_var.val_accuracies,
                                                     train_var.val_accuracy)

        if args.save_convergence:
            train_var.converged = np.append(train_var.converged,
                                            net.converged_samples_per_epoch)
            train_var.not_converged = np.append(train_var.not_converged,
                                                net.not_converged_samples_per_epoch)
            train_var.diverged = np.append(train_var.diverged,
                                           net.diverged_samples_per_epoch)

        if args.compare_with_ndi:
            train_var.rel_dist_to_NDI_vector = np.append(train_var.rel_dist_to_NDI_vector,
                                                         np.mean(net.rel_dist_to_NDI))


        utils.save_summary_dict(args, train_var.summary)

        if net.contains_nans():
            print('Network contains NaNs, terminating training.')
            train_var.summary['finished'] = -1
            break

        if e > 4 and (not args.evaluate):
            if args.dataset in ['mnist', 'fashion_mnist']:
                if train_var.epoch_accuracy < 0.3:
                    print('writing error code -1')
                    train_var.summary['finished'] = -1
                    break
            if args.dataset in ['cifar10']:
                if train_var.epoch_accuracy < 0.25:
                    print('writing error code -1')
                    train_var.summary['finished'] = -1
                    break

            if e > 10 and args.dataset in ['student_teacher']:
                if np.min(train_var.epoch_losses[-10:]) >= np.min(train_var.epoch_losses[:-10]):
                    print('Loss did not improve in the last 10 epochs')
                    train_var.summary['finished'] = -1
                    break

        if e == 2:
            if args.gn_damping_hpsearch:
                print('Doing hpsearch for finding ideal GN damping constant'
                      'for computing the angle with GNT updates')
                gn_damping = gn_damping_hpsearch(args, train_var, device,
                                                 train_loader, net, writer)
                args.gn_damping = gn_damping
                print('Damping constants GNT angles: {}'.format(gn_damping))
                train_var.summary['gn_damping_values'] = gn_damping
                return train_var.summary

    if not args.epochs == 0:
        train_var.summary['avg_time_per_epoch'] = np.mean(train_var.epoch_times)
        train_var.summary['loss_train_last'] = train_var.epoch_loss
        train_var.summary['loss_test_last'] = train_var.test_loss
        train_var.summary['loss_train_best'] = train_var.epoch_losses.min()
        train_var.summary['loss_test_best'] = train_var.test_losses.min()
        train_var.summary['loss_train'] = train_var.epoch_losses
        train_var.summary['loss_test'] = train_var.test_losses
        if not args.no_val_set:
            train_var.summary['loss_val_last'] = train_var.val_loss
            train_var.summary['loss_val_best'] = train_var.val_losses.min()
            train_var.summary['loss_val'] = train_var.val_losses

        if args.save_convergence:
            train_var.summary['converged'] = train_var.converged
            train_var.summary['not_converged'] = train_var.not_converged
            train_var.summary['diverged'] = train_var.diverged

        if args.compare_with_ndi:
            train_var.summary['dist_to_NDI'] = train_var.rel_dist_to_NDI_vector
        else:
            train_var.summary['dist_to_NDI'] = np.array([0])

        if args.save_eigenvalues:
            train_var.summary['max_eig_mean'] = train_var.max_eig_mean_vector
            train_var.summary['max_eig_std'] = train_var.max_eig_std_vector
        else:
            train_var.summary['max_eig_mean'] = np.array([0])
            train_var.summary['max_eig_std'] = np.array([0])
        if args.save_eigenvalues_bcn:
            train_var.summary['max_eig_bcn_mean'] = train_var.max_eig_bcn_mean_vector
            train_var.summary['max_eig_bcn_std'] = train_var.max_eig_bcn_std_vector
        else:
            train_var.summary['max_eig_bcn_mean'] = np.array([0])
            train_var.summary['max_eig_bcn_std'] = np.array([0])

        if args.save_norm_r:
            train_var.summary['norm_r_mean'] = train_var.norm_r_mean_vector
            train_var.summary['norm_r_std'] = train_var.norm_r_std_vector
            train_var.summary['dev_r_mean'] = train_var.dev_r_mean_vector
            train_var.summary['dev_r_std'] = train_var.dev_r_std_vector
        else:
            train_var.summary['norm_r_mean'] = np.array([0])
            train_var.summary['norm_r_std'] = np.array([0])
            train_var.summary['dev_r_mean'] = np.array([0])
            train_var.summary['dev_r_std'] = np.array([0])

        if not args.no_val_set:
            best_epoch = train_var.val_losses.argmin()
            train_var.summary['epoch_best_loss'] = best_epoch
            train_var.summary['loss_test_val_best'] = \
                train_var.test_losses[best_epoch]
            train_var.summary['loss_train_val_best'] = \
                train_var.epoch_losses[best_epoch]

        if args.classification:
            train_var.summary['acc_train_last'] = train_var.epoch_accuracy
            train_var.summary['acc_test_last'] = train_var.test_accuracy
            train_var.summary['acc_train_best'] = train_var.epoch_accuracies.max()
            train_var.summary['acc_test_best'] = train_var.test_accuracies.max()
            train_var.summary['acc_train'] = train_var.epoch_accuracies
            train_var.summary['acc_test'] = train_var.test_accuracies
            if not args.no_val_set:
                train_var.summary['acc_val'] = train_var.val_accuracies
                train_var.summary['acc_val_last'] = train_var.val_accuracy
                train_var.summary['acc_val_best'] = train_var.val_accuracies.max()
                best_epoch = train_var.val_accuracies.argmax()
                train_var.summary['epoch_best_acc'] = best_epoch
                train_var.summary['acc_test_val_best'] = \
                    train_var.test_accuracies[best_epoch]
                train_var.summary['acc_train_val_best'] = \
                    train_var.epoch_accuracies[best_epoch]

        train_var.summary['log_interval'] = args.log_interval
        if args.save_eigenvalues:
            train_var.summary['max_eig_keys'] = [k for k in net.max_eig.keys()]
        elif args.save_eigenvalues_bcn:
            train_var.summary['max_eig_keys'] = [k for k in net.max_eig_bcn.keys()]
        if args.save_norm_r:
            train_var.summary['norm_r_keys'] = [k for k in net.norm_r.keys()]

    utils.save_summary_dict(args, train_var.summary)

    print('Training network ... Done')
    return train_var.summary

def train_separate(args, train_var, device, train_loader, net, writer):
    """
    Train the given network on the given training dataset with DTP. For each
    epoch, first the feedback weights are trained on the whole epoch, after
    which the forward weights are trained on the same epoch (similar to Lee2105)
    Args:
        args (Namespace): The command-line arguments.
        train_var (Namespace): Structure containing training variables
        device: The PyTorch device to be used
        train_loader (torch.utils.data.DataLoader): The data handler for
            training data
        net (LeeDTPNetwork): The neural network
        writer (SummaryWriter): TensorboardX summary writer to save logs
        test_loader (DataLoader): The data handler for the test data
    """

    if not args.freeze_fb_weights:
        for i, (inputs, targets) in enumerate(train_loader):
            if args.dataset not in ['mnist', 'fashion_mnist', 'mnist_autoencoder']:
                inputs, targets = inputs.to(device), targets.to(device)
            if not args.network_type == 'DDTPConv':
                inputs = inputs.flatten(1, -1)
            if args.autoencoder:
                targets = inputs.detach()

            predictions = net.forward(inputs)

            train_feedback_parameters(args, net, train_var.feedback_optimizer)

            if (args.save_logs or args.save_df) and i % args.log_interval == 0:
                if args.save_condition_gn or args.save_jac_t_angle or args.save_jac_pinv_angle:
                    utils.save_feedback_batch_logs(args, writer,
                                                   train_var.batch_idx_fb, net,
                                                   init=False)

                train_var.batch_idx_fb += 1

    for i, (inputs, targets) in enumerate(train_loader):
        if args.dataset not in ['mnist', 'fashion_mnist', 'mnist_autoencoder']:
            inputs, targets = inputs.to(device), targets.to(device)
        if not args.network_type == 'DDTPConv':
            inputs = inputs.flatten(1, -1)
        if args.autoencoder:
            targets = inputs.detach()

        predictions = net.forward(inputs)

        if i % args.log_interval == 0:

            if args.save_eigenvalues:       net.save_eigenvalues = True
            else:                           net.save_eigenvalues = False

            if args.save_eigenvalues_bcn:   net.save_eigenvalues_bcn = True
            else:                           net.save_eigenvalues_bcn = False

            if args.save_norm_r:            net.save_norm_r = True
            else:                           net.save_norm_r = False

            if args.save_NDI_angle:         net.save_NDI_updates = True
            else:                           net.save_NDI_updates = False

        else:
            net.save_eigenvalues = False
            net.save_eigenvalues_bcn = False
            net.save_norm_r = False
            net.save_NDI_updates = False


        train_var.batch_accuracy, train_var.batch_loss = \
            train_forward_parameters(args, net, predictions, targets,
                                     train_var.loss_function,
                                     train_var.forward_optimizer)

        if args.classification:
            train_var.accuracies = np.append(train_var.accuracies,
                                             train_var.batch_accuracy)
        train_var.losses = np.append(train_var.losses,
                                     train_var.batch_loss.item())

        if i % args.log_interval == 0:
            if args.save_logs or args.save_df:
                if args.compute_angles:
                    utils.save_angles(args, writer, train_var.batch_idx, net, train_var.batch_loss, predictions)
                if args.save_eigenvalues:
                    net.save_eigenvalues_to_tensorboard(writer, train_var.batch_idx)
                if args.save_eigenvalues_bcn:
                    net.save_eigenvalues_bcn_to_tensorboard(writer, train_var.batch_idx)
                if args.save_norm_r and args.save_logs:
                    net.save_norm_r_to_tensorboard(writer, train_var.batch_idx)

                if args.save_condition_gn and args.freeze_fb_weights:
                    utils.save_feedback_batch_logs(args, writer,
                                                   train_var.batch_idx, net,
                                                   init=False)

            if args.save_eigenvalues:
                train_var.max_eig_mean_vector, train_var.max_eig_std_vector = \
                    summary_vector_append_mean_std(net.max_eig,
                                                   train_var.max_eig_mean_vector,
                                                   train_var.max_eig_std_vector)

            if args.save_eigenvalues_bcn:
                train_var.max_eig_bcn_mean_vector, train_var.max_eig_bcn_std_vector = \
                    summary_vector_append_mean_std(net.max_eig_bcn,
                                                   train_var.max_eig_bcn_mean_vector,
                                                   train_var.max_eig_bcn_std_vector)

            if args.save_norm_r:
                train_var.norm_r_mean_vector, train_var.norm_r_std_vector = \
                    summary_vector_append_mean_std(net.norm_r,
                                                   train_var.norm_r_mean_vector,
                                                   train_var.norm_r_std_vector)
                train_var.dev_r_mean_vector = np.append(train_var.dev_r_mean_vector,
                                                        np.mean(net.dev_r))
                train_var.dev_r_std_vector = np.append(train_var.dev_r_std_vector,
                                                       np.std(net.dev_r))

            train_var.batch_idx += 1

        if not args.freeze_forward_weights:
            train_var.forward_optimizer.step()


def summary_vector_append_mean_std(batch_dict, vector_mean, vector_std):
    """
    Takes the results from one batch (of max eig or norm_r)
    and appends n values of mean and std (where n is the number
    of keys in batch dict).
    :param: batch_dict : dictionary containing results for one metric for one batch
    :param: vector_mean: vector of means to which the means of batch_dict data will be appendend
    :param: vector_std: vector of means to which the standard deviations of batch_dict data will be appendend
    :return: `vector_mean`, `vector_std` with the new values appended
    """

    keys = [k for k in batch_dict.keys()]
    mean = np.zeros((len(keys), 1))
    std = np.zeros((len(keys), 1))
    for i, k in enumerate(keys):
        mean[i] = np.mean(batch_dict[k])
        std[i] = np.std(batch_dict[k])
    vector_mean = np.append(vector_mean, mean)
    vector_std = np.append(vector_std, std)
    return vector_mean, vector_std


def train_forward_parameters(args, net, predictions, targets, loss_function,
                                 forward_optimizer):
    """
    Train the forward parameters on the current mini-batch.
    """
    if predictions.requires_grad == False:
        predictions.requires_grad = True

    save_target = args.compute_angles
    forward_optimizer.zero_grad()
    loss = loss_function(predictions, targets)

    net.backward(loss, args.target_stepsize, save_target=save_target)

    if args.classification:
        batch_accuracy = utils.accuracy(predictions, targets)
    else:
        batch_accuracy = None
    batch_loss = loss

    if args.fix_grad_norm > 0:
        utils.fix_grad_norm_(forward_optimizer.parameters,
                             fixed_norm=args.fix_grad_norm,
                             norm_type=2.)

    return batch_accuracy, batch_loss


def train_feedback_parameters(args, net, feedback_optimizer):
    """
    Train the feedback parameters on the current mini-batch.
    """

    feedback_optimizer.zero_grad()

    net.compute_feedback_gradients()

    if net.clip_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(net.get_feedback_parameter_list(),
                                       max_norm=net.clip_grad_norm)

    feedback_optimizer.step()


def test(args, device, net, test_loader, loss_function, train_var=None):
    """
    Compute the test loss and accuracy on the test dataset
    Args:
        args: command line inputs
        net: network
        test_loader (DataLoader): dataloader object with the test dataset
    Returns: Tuple containing:
        - Test accuracy
        - Test loss
    """

    loss = 0
    if args.classification:
        accuracy = 0
    nb_batches = len(test_loader)
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):

            if args.dataset not in ['mnist', 'fashion_mnist', 'mnist_autoencoder']:
                inputs, targets = inputs.to(device), targets.to(device)
            if not args.network_type == 'DDTPConv':
                inputs = inputs.flatten(1, -1)

            if args.autoencoder:
                targets = inputs.detach()

            predictions = net.forward(inputs)
            loss += loss_function(predictions, targets).item()
            if args.classification:
                accuracy += utils.accuracy(predictions, targets)

            if args.plot_autoencoder_images and i == 0:
                input_images = torch.reshape(inputs,
                                             (args.batch_size, 1, 28, 28))[0:5]
                reconstructed_images = torch.reshape(predictions, (
                    args.batch_size, 1, 28, 28))[0:5]
                save_image(input_images, os.path.join(
                    args.out_dir,
                    'autoencoder_images/input_epoch_{}.png'.format(
                        train_var.epochs)))
                save_image(reconstructed_images, os.path.join(
                    args.out_dir,
                    'autoencoder_images/reconstructed_epoch_{}.png'.format(
                        train_var.epochs)))
    loss /= nb_batches
    if args.classification:
        accuracy /= nb_batches
    else:
        accuracy = None
    return accuracy, loss


def train_only_feedback_parameters(args, train_var, device, train_loader,
                                       net, writer, log=True,
                                   compute_gn_condition=False, init=False):
    """
    Train only the feedback parameters for the given amount of epochs.
    This function is used to initialize the network in a 'pseudo-inverse'
    condition.
    """

    for i, (inputs, targets) in enumerate(train_loader):
        if args.dataset not in ['mnist', 'fashion_mnist', 'mnist_autoencoder']:
            inputs, targets = inputs.to(device), targets.to(device)
        if not args.network_type == 'DDTPConv':
            inputs = inputs.flatten(1, -1)

        predictions = net.forward(inputs)

        if init:
            train_feedback_parameters(args, net, train_var.feedback_init_optimizer)
        else:
            train_feedback_parameters(args, net, train_var.feedback_optimizer)

        if (args.save_logs or args.save_df) and i % args.log_interval == 0:
            if args.save_condition_gn or args.save_jac_t_angle or args.save_jac_pinv_angle:
                if init:
                    utils.save_feedback_batch_logs(args, writer,
                                           train_var.init_idx, net,
                                           init=init,
                                        statistics=args.save_fb_statistics_init)
                    train_var.init_idx += 1

        if compute_gn_condition and i % args.log_interval == 0:
            gn_condition = net.compute_condition_two(retain_graph=False)
            train_var.gn_condition_init = np.append(train_var.gn_condition_init,
                                                    gn_condition.item())


def train_extra_fb_minibatches(args, train_var, device, train_loader,
                                   net):
    train_loader_iter = iter(train_loader)
    for i in range(args.extra_fb_minibatches):
        if not args.network_type == 'DDTPConv':
            inputs = inputs.flatten(1, -1)
        predictions = net.forward(inputs)
        train_feedback_parameters(args, net, train_var.feedback_optimizer)


def train_bp(args, device, train_loader, net, writer, test_loader, summary,
                 val_loader):
    print('Training network ...')
    net.train()
    forward_optimizer = utils.OptimizerList(args, net)

    nb_batches = len(train_loader)

    if args.classification:
        if args.output_activation == 'softmax':
            loss_function = nn.CrossEntropyLoss()
        else:
            raise ValueError('The mnist dataset can only be combined with a '
                             'softmax output activation.')

    elif args.regression or args.autoencoder:
        loss_function = nn.MSELoss()
    else:
        raise ValueError('The provided dataset {} is not supported.'.format(
            args.dataset
        ))

    epoch_losses = np.array([])
    test_losses = np.array([])
    val_losses = np.array([])
    val_loss = None
    val_accuracy = None

    if args.classification:
        epoch_accuracies = np.array([])
        test_accuracies = np.array([])
        val_accuracies = np.array([])

    for e in range(args.epochs):
        if args.classification:
            running_accuracy = 0
        else:
            running_accuracy = None
        running_loss = 0
        for i, (inputs, targets) in enumerate(train_loader):
            if args.dataset not in ['mnist', 'fashion_mnist',
                                    'mnist_autoencoder']:
                inputs, targets = inputs.to(device), targets.to(device)
            if not args.network_type == 'BPConv':
                inputs = inputs.flatten(1, -1)

            if args.autoencoder:
                targets = inputs.detach()

            forward_optimizer.zero_grad()
            predictions = net(inputs)
            loss = loss_function(predictions, targets)
            loss.backward()

            if args.fix_grad_norm > 0:
                utils.fix_grad_norm_(net.parameters(),
                                     fixed_norm=args.fix_grad_norm,
                                     norm_type=2.)

            forward_optimizer.step()

            running_loss += loss.item()

            if args.classification:
                running_accuracy += utils.accuracy(predictions, targets)

        if not args.no_val_set:
            val_accuracy, val_loss = test_bp(args, device, net, val_loader,
                                                 loss_function, epoch=e)

        test_accuracy, test_loss = test_bp(args, device, net, test_loader,
                                               loss_function, epoch=e)

        epoch_loss = running_loss/nb_batches
        if args.classification:
            epoch_accuracy = running_accuracy/nb_batches
        else:
            epoch_accuracy = None

        print('Epoch {} '.format(e + 1))
        print('\ttraining loss = {}'.format(np.round(epoch_loss,6)))
        if not args.no_val_set:
            print('\tval loss = {}.'.format(np.round(val_loss,6)))
        print('\ttest loss = {}.'.format(np.round(test_loss,6)))

        if args.classification:
            print('\ttraining acc  = {}%'.format(np.round(epoch_accuracy * 100,6)))
            if not args.no_val_set:
                print('\tval acc  = {}%'.format(np.round(val_accuracy * 100,6)))
            print('\ttest acc  = {}%'.format(np.round(test_accuracy * 100,6)))

        if args.save_logs:
            utils.save_logs(writer, step=e + 1, net=net,
                            loss=epoch_loss,
                            epoch_time=2.,
                            accuracy=epoch_accuracy,
                            test_loss=test_loss,
                            test_accuracy=test_accuracy,
                            val_loss=val_loss,
                            val_accuracy=val_accuracy)

        epoch_losses = np.append(epoch_losses,
                                           epoch_loss)
        test_losses = np.append(test_losses,
                                          test_loss)
        if not args.no_val_set:
            val_losses = np.append(val_losses, val_loss)

        if args.classification:
            epoch_accuracies = np.append(
                epoch_accuracies,
                epoch_accuracy)
            test_accuracies = np.append(test_accuracies,
                                                  test_accuracy)
            if not args.no_val_set:
                val_accuracies = np.append(val_accuracies, val_accuracy)

        utils.save_summary_dict(args, summary)

        if e > 4:
            if args.dataset in ['mnist', 'fashion_mnist']:
                if epoch_accuracy < 0.4:
                    print('writing error code -1')
                    summary['finished'] = -1
                    break
            if args.dataset in ['cifar10']:
                if epoch_accuracy < 0.25:
                    print('writing error code -1')
                    summary['finished'] = -1
                    break

    if not args.epochs == 0:
        summary['loss_train_last'] = epoch_loss
        summary['loss_test_last'] = test_loss
        summary['loss_train_best'] = epoch_losses.min()
        summary['loss_test_best'] = test_losses.min()
        summary['loss_train'] = epoch_losses
        summary['loss_test'] = test_losses
        if not args.no_val_set:
            summary['loss_val_last'] = val_loss
            summary['loss_val_best'] = val_losses.min()
            summary['loss_val'] = val_losses
            best_epoch = val_losses.argmin()
            summary['epoch_best_loss'] = best_epoch
            summary['loss_test_val_best'] = \
                test_losses[best_epoch]
            summary['loss_train_val_best'] = \
                epoch_losses[best_epoch]

        if args.classification:
            summary['acc_train_last'] = epoch_accuracy
            summary['acc_test_last'] = test_accuracy
            summary[
                'acc_train_best'] = epoch_accuracies.max()
            summary[
                'acc_test_best'] = test_accuracies.max()
            summary['acc_train'] = epoch_accuracies
            summary['acc_test'] = test_accuracies
            if not args.no_val_set:
                summary['acc_val'] = val_accuracies
                summary['acc_val_last'] = val_accuracy
                summary['acc_val_best'] = val_accuracies.max()
                best_epoch = val_accuracies.argmax()
                summary['epoch_best_acc'] = best_epoch
                summary['acc_test_val_best'] = \
                    test_accuracies[best_epoch]
                summary['acc_train_val_best'] = \
                    epoch_accuracies[best_epoch]
    utils.save_summary_dict(args, summary)

    print('Training network ... Done')
    return summary


def test_bp(args, device, net, test_loader, loss_function, epoch):
    loss = 0
    if args.classification:
        accuracy = 0
    nb_batches = len(test_loader)
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            if args.dataset not in ['mnist', 'fashion_mnist',
                                    'mnist_autoencoder']:
                inputs, targets = inputs.to(device), targets.to(device)
            if not args.network_type == 'BPConv':
                inputs = inputs.flatten(1, -1)
            if args.autoencoder:
                targets = inputs.detach()
            predictions = net(inputs)
            loss += loss_function(predictions, targets).item()
            if args.classification:
                accuracy += utils.accuracy(predictions, targets)

            if args.plot_autoencoder_images and i == 0:
                input_images = torch.reshape(inputs,
                                             (args.batch_size,1, 28, 28))[0:5]
                reconstructed_images = torch.reshape(predictions, (
                    args.batch_size,1, 28, 28))[0:5]
                save_image(input_images, os.path.join(
                    args.out_dir,
                    'autoencoder_images/input_epoch_{}.png'.format(
                        epoch)))
                save_image(reconstructed_images, os.path.join(
                    args.out_dir,
                    'autoencoder_images/reconstructed_epoch_{}.png'.format(
                        epoch)))
    loss /= nb_batches
    if args.classification:
        accuracy /= nb_batches
    else:
        accuracy = None
    return accuracy, loss


def gn_damping_hpsearch(args, train_var, device, train_loader, net, writer):
    freeze_forward_weights_copy = args.freeze_forward_weights
    args.freeze_forward_weights = True
    damping_values = np.logspace(-5., 1., num=7, base=10.0)
    damping_values = np.append(0, damping_values)
    average_angles = np.empty((len(damping_values), net.depth))

    for k, gn_damping in enumerate(damping_values):
        print('testing damping={}'.format(gn_damping))
        angles_df = pd.DataFrame(columns=[i for i in range(0, net.depth)])
        step=0
        for i, (inputs, targets) in enumerate(train_loader):
            if args.dataset not in ['mnist', 'fashion_mnist',
                                    'mnist_autoencoder']:
                inputs, targets = inputs.to(device), targets.to(device)
            if not args.network_type == 'DDTPConv':
                inputs = inputs.flatten(1, -1)
            if args.autoencoder:
                targets = inputs.detach()

            predictions = net.forward(inputs)

            acc, loss = \
                train_forward_parameters(args, net, predictions, targets,
                                             train_var.loss_function,
                                             train_var.forward_optimizer)

            if  i % args.log_interval == 0:
                net.save_gnt_angles(writer, step, predictions,
                                    loss, gn_damping,
                                    retain_graph=False,
                                    custom_result_df=angles_df)
                step += 1

        average_angles[k, :] = angles_df.mean(axis=0)

    optimal_damping_constants_layerwise = damping_values[average_angles.argmin(axis=0)]
    optimal_damping_constant = damping_values[average_angles.mean(axis=1).argmin(axis=0)]
    print('average angles:')
    print(average_angles)
    print('optimal damping constant: {}'.format(optimal_damping_constant))
    print('optimal damping constants layerwise: {}'.format(optimal_damping_constants_layerwise))

    file_path = os.path.join(args.out_dir, 'optimal_gnt_damping_constant.txt')
    with open(file_path, 'w') as f:
        f.write('average angles:\n')
        f.write(str(average_angles) + '\n')
        f.write('optimal damping constant: {} \n'.format(optimal_damping_constant))
        f.write('optimal damping constants layerwise: {} \n'.format(optimal_damping_constants_layerwise))

    args.freeze_forward_weights = freeze_forward_weights_copy

    return optimal_damping_constant