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
A collection of helper functions
--------------------------------
"""
 
import numpy as np
import torch
import time
from torch.utils.data import Dataset
from tensorboardX import SummaryWriter
import os
import pandas
import warnings
import matplotlib
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision.datasets import FashionMNIST
matplotlib.use('Agg')


class RegressionDataset(Dataset):
    """
    A simple regression dataset.
    Args:
        inputs (numpy.ndarray): The input samples.
        outputs (numpy.ndarray): The output samples.
    """

    def __init__(self, inputs, outputs, double_precision=False):
        assert(len(inputs.shape) == 2)
        assert(len(outputs.shape) == 2)
        assert(inputs.shape[0] == outputs.shape[0])

        if double_precision:
            self.inputs = torch.from_numpy(inputs).to(torch.float64)
            self.outputs = torch.from_numpy(outputs).to(torch.float64)
        else:
            self.inputs = torch.from_numpy(inputs).float()
            self.outputs = torch.from_numpy(outputs).float()


    def __len__(self):
        return int(self.inputs.shape[0])


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        batch_in = self.inputs[idx, :]
        batch_out = self.outputs[idx, :]

        return batch_in, batch_out


def accuracy(predictions, labels):
    """
    Compute the average accuracy of the given predictions.
    Inspired on
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    Args:
        predictions (torch.Tensor): Tensor containing the output of the linear
            output layer of the network.
        labels (torch.Tensor): Tensor containing the labels of the mini-batch
    Returns (float): average accuracy of the given predictions
    """

    _, pred_labels = torch.max(predictions.data, 1)
    total = labels.size(0)
    correct = (pred_labels == labels).sum().item()

    return correct/total


def choose_optimizer(args, net):
    """
    Return the wished optimizer (based on inputs from args).
    Args:
        args: cli
        net: neural network
    Returns: optimizer
    """

    forward_optimizer = OptimizerList(args, net)
    # feedback_optimizer = choose_feedback_optimizer(args, net)
    feedback_optimizer = FbOptimizerList(args, net, args.lr_fb)
    feedback_init_optimizer = FbOptimizerList(args, net, args.lr_fb_init)

    return forward_optimizer, feedback_optimizer, feedback_init_optimizer


def choose_forward_optimizer(args, net):
    """
    Return the wished optimizer (based on inputs from args).
    Args:
        args: cli
        net: neural network
    Returns: optimizer
    """

    if args.network_type in ('BP'):
        if args.shallow_training:
            print('Shallow training')
            forward_params = net.layers[-1].parameters()
        elif args.only_train_first_layer:
            print('Only training first layer')
            forward_params = net.layers[0].parameters()
        else:
            forward_params = net.parameters()
    else:
        if args.only_train_first_layer:
            print('Only training first layer')
            forward_params = net.get_forward_parameter_list_first_layer()
        else:
            forward_params = net.get_forward_parameter_list()

    if args.optimizer == 'SGD':
        print('Using SGD optimizer')
        forward_optimizer = torch.optim.SGD(forward_params,
                                            lr=args.lr, momentum=args.momentum,
                                            weight_decay=args.forward_wd)
    elif args.optimizer == 'RMSprop':
        print('Using RMSprop optimizer')

        forward_optimizer = torch.optim.RMSprop(
            forward_params,
            lr=args.lr,
            momentum=args.momentum,
            alpha=0.95,
            eps=0.03,
            weight_decay=args.forward_wd,
            centered=True
        )

    elif args.optimizer == 'Adam':
        print('Using Adam optimizer.')

        forward_optimizer = torch.optim.Adam(
            forward_params,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.epsilon,
            weight_decay=args.forward_wd
        )

    else:
        raise ValueError('Provided optimizer "{}" is not supported'.format(
            args.optimizer
        ))

    return forward_optimizer


def choose_feedback_optimizer(args, net, lr_fb):
    """
    Return the wished optimizer (based on inputs from args).
    Args:
        args: cli
        net: neural network
    Returns: optimizer
    """

    if args.freeze_fb_weights_output:
        feedback_params = net.get_feedback_parameter_list()[:-1]
    else:
        feedback_params = net.get_feedback_parameter_list()

    if args.optimizer_fb == 'SGD':
        feedback_optimizer = torch.optim.SGD(feedback_params,
                                             lr=lr_fb,
                                             weight_decay=args.feedback_wd)
    elif args.optimizer_fb == 'RMSprop':

        feedback_optimizer = torch.optim.RMSprop(
            feedback_params,
            lr=lr_fb,
            momentum=args.momentum,
            alpha=0.95,
            eps=0.03,
            weight_decay=args.feedback_wd,
            centered=True
        )

    elif args.optimizer_fb == 'Adam':

        feedback_optimizer = torch.optim.Adam(
            feedback_params,
            lr=lr_fb,
            betas=(args.beta1_fb, args.beta2_fb),
            eps=args.epsilon_fb,
            weight_decay=args.feedback_wd
        )

    else:
        raise ValueError('Provided optimizer "{}" is not supported'.format(
            args.optimizer
        ))

    return feedback_optimizer


class OptimizerList(object):
    """ A class for stacking a separate optimizer for each layer in a list. If
    no separate learning rates per layer are required, a single optimizer is
    stored in the optimizer list."""

    def __init__(self, args, net):
        if isinstance(args.lr, float):
            forward_optimizer = choose_forward_optimizer(args, net)
            optimizer_list = [forward_optimizer]
        elif isinstance(args.lr, np.ndarray):

            if args.only_train_first_layer:
                print('Only training first layer')
                forward_params = \
                    net.get_forward_parameter_list_first_layer()
            elif args.freeze_output_layer:
                print('Freezing output layer')
                forward_params = net.get_reduced_forward_parameter_list()
            else:
                forward_params = net.get_forward_parameter_list()
            if not (args.optimizer == 'SGD' or args.optimizer == 'Adam'):
                raise NetworkError('multiple learning rates are only supported '
                                   'for SGD optimizer')

            optimizer_list = []
            for i, lr in enumerate(args.lr):
                eps = args.epsilon[i]
                if args.network_type == 'BPConv':
                    if i == 0:
                        j = 0
                    elif i == 1:
                        j = 2
                    elif i == 2:
                        j = 4
                    elif i == 3:
                        j = 5
                    if args.no_bias:
                        parameters = [net.layers[j].weight]
                    else:
                        parameters = [net.layers[j].weight, net.layers[j].bias]
                else:
                    if args.no_bias:
                        parameters = [net.layers[i].weights]
                    else:
                        parameters = [net.layers[i].weights, net.layers[i].bias]
                if args.optimizer == 'SGD':
                    optimizer = torch.optim.SGD(parameters,
                                                lr=lr, momentum=args.momentum,
                                                weight_decay=args.forward_wd)
                elif args.optimizer == 'Adam':
                    optimizer = torch.optim.Adam(
                        parameters,
                        lr=lr,
                        betas=(args.beta1, args.beta2),
                        eps=eps,
                        weight_decay=args.forward_wd)
                optimizer_list.append(optimizer)
        else:
            raise ValueError('Command line argument lr={} is not recognized '
                             'as a float'
                       'or list'.format(args.lr))

        self._optimizer_list = optimizer_list

        if args.network_type in ('BP'):
            if args.shallow_training:
                print('Shallow training')
                forward_params = net.layers[-1].parameters()
            elif args.only_train_first_layer:
                print('Only training first layer')
                forward_params = net.layers[0].parameters()
            else:
                forward_params = net.parameters()
        else:
            if args.only_train_first_layer:
                print('Only training first layer')
                forward_params = net.get_forward_parameter_list_first_layer()
            else:
                forward_params = net.get_forward_parameter_list()
        self.parameters = forward_params


    def zero_grad(self):
        for optimizer in self._optimizer_list:
            optimizer.zero_grad()

    def step(self, i=None):
        """
        Perform a step on the optimizer of layer i. If i is None, a step is
        performed on all optimizers.
        """
        if i is None:
            for optimizer in self._optimizer_list:
                optimizer.step()
        else:
            self._optimizer_list[i].step()


class FbOptimizerList(object):
    def __init__(self, args, net, lr_fb):
        
        if isinstance(lr_fb, float):
            fb_optimizer = choose_feedback_optimizer(args, net, lr_fb)
            optimizer_list = [fb_optimizer]
        else:
            raise NotImplementedError("Different learning rates for feedback"
                                      "weights is not implemented yet.")

        self._optimizer_list = optimizer_list
        if args.freeze_fb_weights_output:
            feedback_params = net.get_feedback_parameter_list()[:-1]
        else:
            feedback_params = net.get_feedback_parameter_list()
        self.parameters = feedback_params

    def step(self):
        for optimizer in self._optimizer_list:
            optimizer.step()

    def zero_grad(self):
        for optimizer in self._optimizer_list:
            optimizer.zero_grad()


def save_logs(writer, step, net, loss, epoch_time, accuracy, test_loss, test_accuracy,
              val_loss, val_accuracy):
    """
    Save logs and plots to tensorboardX
    Args:
        writer (SummaryWriter): TensorboardX summary writer
        step: global step
        net: network
        loss: current loss of the training iteration
    """

    net.save_logs(writer, step)
    writer.add_scalar(tag='training_metrics/epoch_time',
                      scalar_value=epoch_time,
                      global_step=step)
    writer.add_scalar(tag='training_metrics/loss',
                      scalar_value=loss,
                      global_step=step)
    writer.add_scalar(tag='training_metrics/test_loss',
                      scalar_value=test_loss,
                      global_step=step)
    if val_loss is not None:
        writer.add_scalar(tag='training_metrics/val_loss',
                          scalar_value=val_loss,
                          global_step=step)
    if accuracy is not None:
        writer.add_scalar(tag='training_metrics/accuracy',
                          scalar_value=accuracy,
                          global_step=step)
        writer.add_scalar(tag='training_metrics/test_accuracy',
                          scalar_value=test_accuracy,
                          global_step=step)
        if val_accuracy is not None:
            writer.add_scalar(tag='training_metrics/val_accuracy',
                              scalar_value=val_accuracy,
                              global_step=step)


def save_logs_convergence(writer, step, net, loss, epoch_time, accuracy, test_loss, test_accuracy,
                          val_loss, val_accuracy,
                          converged_samples_per_epoch, diverged_samples_per_epoch, not_converged_samples_per_epoch,
                          dist_to_NDI):
    """
    Save logs and plots to tensorboardX
    Args:
        writer (SummaryWriter): TensorboardX summary writer
        step: global step
        net: network
        loss: current loss of the training iteration
    """

    net.save_logs(writer, step)
    writer.add_scalar(tag='training_metrics/loss',
                      scalar_value=loss,
                      global_step=step)
    writer.add_scalar(tag='epoch_time',
                      scalar_value=epoch_time,
                      global_step=step)
    writer.add_scalar(tag='training_metrics/test_loss',
                      scalar_value=test_loss,
                      global_step=step)
    if val_loss is not None:
        writer.add_scalar(tag='training_metrics/val_loss',
                          scalar_value=val_loss,
                          global_step=step)
    if accuracy is not None:
        writer.add_scalar(tag='training_metrics/accuracy',
                          scalar_value=accuracy,
                          global_step=step)
        writer.add_scalar(tag='training_metrics/test_accuracy',
                          scalar_value=test_accuracy,
                          global_step=step)
        if val_accuracy is not None:
            writer.add_scalar(tag='training_metrics/val_accuracy',
                              scalar_value=val_accuracy,
                              global_step=step)

    writer.add_scalar(tag='sample_convergence/converged_samples',
                      scalar_value=converged_samples_per_epoch,
                      global_step=step)
    writer.add_scalar(tag='sample_convergence/diverged_samples',
                      scalar_value=diverged_samples_per_epoch,
                      global_step=step)
    writer.add_scalar(tag='sample_convergence/not_converged_samples',
                      scalar_value=not_converged_samples_per_epoch,
                      global_step=step)
    if dist_to_NDI is not None:
        writer.add_scalar(tag='sample_convergence/rel_dist_to_NDI',
                          scalar_value=dist_to_NDI,
                          global_step=step)


def save_angles(args, writer, step, net, loss, output_activation):
    """
    Save logs and plots for the current mini-batch on tensorboardX
    Args:
        args (Namespace): commandline arguments
        writer (SummaryWriter): TensorboardX summary writer
        step: global step
        net (networks.DTPNetwork): network
        loss (torch.Tensor): loss of the current minibatch
        output_activation (torch.Tensor): output of the network for the current
            minibatch
    """

    if args.save_BP_angle:
        retain_graph = args.save_GN_angle or args.save_GNT_angle or \
                       args.save_nullspace_norm_ratio or \
                       args.save_GN_GNT_angle or args.save_BP_GNT_angle or \
                       args.save_GNT_ss_no_ss_angle
        net.save_bp_angles(writer, step, loss,
                           retain_graph=retain_graph,
                           save_tensorboard=args.save_logs,
                           save_dataframe=args.save_df)
    if args.save_GN_angle:
        retain_graph = args.save_GNT_angle or \
                       args.save_nullspace_norm_ratio or \
                       args.save_GN_GNT_angle or args.save_BP_GNT_angle or \
                       args.save_GNT_ss_no_ss_angle
        net.save_gn_angles(writer, step, output_activation, loss,
                           args.gn_damping,
                           retain_graph=retain_graph,
                           save_tensorboard=args.save_logs,
                           save_dataframe=args.save_df)
    if args.save_GNT_angle:
        retain_graph = args.save_nullspace_norm_ratio or \
                       args.save_GN_GNT_angle or args.save_BP_GNT_angle or \
                       args.save_GNT_ss_no_ss_angle
        net.save_gnt_angles(writer, step, output_activation, loss,
                            args.gn_damping,
                            retain_graph=retain_graph,
                            save_tensorboard=args.save_logs,
                            save_dataframe=args.save_df,
                            steady_state=args.use_ss_gnt,
                            nonlinear=args.use_nonlinear_gnt
                            )

    if args.save_NDI_angle and args.network_type == 'DFC':
        net.save_ndi_angles(writer, step,
                            save_tensorboard=args.save_logs,
                            save_dataframe=args.save_df
                            )

    if args.save_nullspace_norm_ratio:
        retain_graph = args.save_condition_gn or \
                       args.save_GN_GNT_angle or args.save_BP_GNT_angle or \
                       args.save_GNT_ss_no_ss_angle
        net.save_nullspace_norm_ratio(writer, step, output_activation,
                                  retain_graph,
                           save_tensorboard=args.save_logs,
                           save_dataframe=args.save_df)

    if args.save_BP_GNT_angle:
        retain_graph = args.save_GN_GNT_angle or args.save_condition_gn or \
                       args.save_GNT_ss_no_ss_angle
        net.save_bp_gnt_angles( writer, step, output_activation, loss,
                           args.gn_damping, retain_graph,
                           save_tensorboard=args.save_logs,
                           save_dataframe=args.save_df,
                           steady_state=args.use_ss_gnt,
                           nonlinear=args.use_nonlinear_gnt)

    if args.save_GN_GNT_angle:
        retain_graph = args.save_condition_gn or args.save_GNT_ss_no_ss_angle
        net.save_gn_gnt_angles(writer, step, output_activation, loss,
                               args.gn_damping,
                               retain_graph=retain_graph,
                               save_tensorboard=args.save_logs,
                               save_dataframe=args.save_df,
                               steady_state=args.use_ss_gnt,
                               nonlinear=args.use_nonlinear_gnt)

    if args.save_GNT_ss_no_ss_angle:
        retain_graph = args.save_condition_gn
        net.save_gnt_ss_no_ss_angles(writer, step, output_activation, loss,
                           args.gn_damping, retain_graph=retain_graph,
                           save_tensorboard=args.save_logs,
                           save_dataframe=args.save_df,
                           steady_state=args.use_ss_gnt,
                           nonlinear=args.use_nonlinear_gnt)


def save_feedback_batch_logs(args, writer, step, net, init=False,
                             statistics=False):
    """
    Save logs and plots for the current mini-batch on tensorboardX
    Args:
        args (Namespace): commandline arguments
        writer (SummaryWriter): TensorboardX summary writer
        step: global step
        net (networks.DTPNetwork): network
        init (bool): flag indicating that the training is in the
                initialization phase (only training the feedback weights).
        statistics (bool): Flag indicating whether the statistics of the
            feedback weights should be saved (e.g. norm of the gradients etc).
    """

    net.save_feedback_batch_logs(args, writer, step, init=init,
                                 save_tensorboard=args.save_logs,
                                 save_dataframe=args.save_df,
                                 save_statistics=statistics,
                                 damping=args.gn_damping)


def save_gradient_hook(module, grad_input, grad_output):
    """ A hook that will be used to save the gradients the loss with respect
             to the output of the network. This gradient is used to compute the
              target for the output layer."""
    print('save grad in module')
    module.output_network_gradient = grad_input[0]


def compute_jacobian(input, output, structured_tensor=False,
                     retain_graph=False):
    """
    Compute the Jacobian matrix of output with respect to input. If input
    and/or output have more than one dimension, the Jacobian of the flattened
    output with respect to the flattened input is returned if
    structured_tensor is False. If structured_tensor is True, the Jacobian is
    structured in dimensions output_shape x flattened input shape. Note that
    output_shape can contain multiple dimensions.
    Args:
        input (list or torch.Tensor): Tensor or sequence of tensors
            with the parameters to which the Jacobian should be
            computed. Important: the requires_grad attribute of input needs to
            be True while computing output in the forward pass.
        output (torch.Tensor): Tensor with the values of which the Jacobian is
            computed
        structured_tensor (bool): A flag indicating if the Jacobian
            should be structured in a tensor of shape
            output_shape x flattened input shape instead of
            flattened output shape x flattened input shape.
    Returns (torch.Tensor): 2D tensor containing the Jacobian of output with
        respect to input if structured_tensor is False. If structured_tensor
        is True, the Jacobian is structured in a tensor of shape
        output_shape x flattened input shape.
    """

    if isinstance(input, torch.Tensor):
        input = [input]

    output_flat = output.view(-1)
    numel_input = 0
    for input_tensor in input:
        numel_input += input_tensor.numel()
    jacobian = torch.Tensor(output.numel(), numel_input)

    for i, output_elem in enumerate(output_flat):

        if i == output_flat.numel() - 1:
            gradients = torch.autograd.grad(output_elem, input,
                                            retain_graph=retain_graph,
                                            create_graph=False,
                                            only_inputs=True)
        else:
            gradients = torch.autograd.grad(output_elem, input,
                                            retain_graph=True,
                                            create_graph=False,
                                            only_inputs=True)
        jacobian_row = torch.cat([g.view(-1).detach() for g in gradients])
        jacobian[i, :] = jacobian_row

    if structured_tensor:
        shape = list(output.shape)
        shape.append(-1) 
        jacobian = jacobian.view(shape)

    return jacobian


def compute_batch_jacobian(input, output, retain_graph=False):
    """
    Compute the Jacobian matrix of a batch of outputs with respect to
    some input (normally, the activations of a hidden layer).
    Returned Jacobian has dimensions Batch x SizeOutput x SizeInput
    Args:
        input (list or torch.Tensor): Tensor or sequence of tensors
            with the parameters to which the Jacobian should be
            computed. Important: the requires_grad attribute of input needs to
            be True while computing output in the forward pass.
        output (torch.Tensor): Tensor with the values of which the Jacobian is
            computed
    Returns (torch.Tensor): 3D tensor containing the Jacobian of output with
        respect to input: batch_size x output_size x input_size.
    """

    batch_jacobian = torch.Tensor(output.shape[0], output.shape[1], input.shape[1])
    assert output.shape[0] == input.shape[0], \
        "Batch size needs to be the same for both input and output"

    for batch_idx in range(output.shape[0]):

        for i, output_elem in enumerate(output[batch_idx]):

            if i < output.shape[1]: rg = True
            else: rg = retain_graph
            gradients = torch.autograd.grad(output_elem, input, retain_graph=rg)[0][batch_idx].detach()
            batch_jacobian[batch_idx, i, :] = gradients

    return batch_jacobian


def compute_damped_gn_update(jacobian, output_error, damping):
    """
    Compute the damped Gauss-Newton update, based on the given jacobian and
    output error.
    Args:
        jacobian (torch.Tensor): 2D tensor containing the Jacobian of the
            flattened output with respect to the flattened parameters for which
            the GN update is computed.
        output_error (torch.Tensor): tensor containing the gradient of the loss
            with respect to the output layer of the network.
        damping (float): positive damping hyperparameter
    Returns: the damped Gauss-Newton update for the parameters for which the
        jacobian was computed.
    """

    if damping < 0:
        raise ValueError('Positive value for damping expected, got '
                         '{}'.format(damping))
    output_error = output_error.view(-1, 1).detach()

    if damping == 0:
        jacobian_pinv = torch.pinverse(jacobian)
        gn_updates = jacobian_pinv.mm(output_error)
    else:
        if jacobian.shape[0] >= jacobian.shape[1]:
            G = jacobian.t().mm(jacobian)
            C = G + damping * torch.eye(G.shape[0])
            C_cholesky = torch.cholesky(C)
            jacobian_error = jacobian.t().matmul(output_error)
            gn_updates = torch.cholesky_solve(jacobian_error, C_cholesky)
        else:
            G = jacobian.mm(jacobian.t())
            C = G + damping * torch.eye(G.shape[0])
            C_cholesky = torch.cholesky(C)
            inverse_error = torch.cholesky_solve(output_error, C_cholesky)
            gn_updates = jacobian.t().matmul(inverse_error)

    return gn_updates


def compute_angle(A, B):
    """
    Compute the angle between two tensors of the same size. The tensors will
     be flattened, after which the angle is computed.
    Args:
        A (torch.Tensor): First tensor
        B (torch.Tensor): Second tensor
    Returns: The angle between the two tensors in degrees
    """
    if contains_nan(A):
        print('tensor A contains nans:')
        print(A)
    if contains_nan(B):
        print('tensor B contains nans:')
        print(B)

    inner_product = torch.sum(A*B)  #equal to inner product of flattened tensors
    cosine = inner_product/(torch.norm(A, p='fro')*torch.norm(B, p='fro'))

    if cosine > 1 and cosine < 1 + 1e-5:
        cosine = torch.Tensor([1.])
    angle = 180/np.pi*torch.acos(cosine)
    if contains_nan(angle):
        print('angle computation causes NANs. cosines:')
        print(cosine)
    return angle


def compute_average_batch_angle(A, B):
    """
    Compute the average of the angles between the mini-batch samples of A and B.
    If the samples of the mini-batch have more than one dimension (minibatch
    dimension not included), the tensors will first be flattened
    Args:
        A (torch.Tensor):  A tensor with as first dimension the mini-batch
            dimension
        B (torch.Tensor): A tensor of the same shape as A
    Returns: The average angle between the two tensors in degrees.
    """

    A = A.flatten(1, -1)
    B = B.flatten(1, -1)
    if contains_nan(A):
        print('tensor A contains nans in activation angles:')
    if contains_nan(B):
        print('tensor B contains nans in activation angles:')
    inner_products = torch.sum(A*B, dim=1)
    A_norms = torch.norm(A, p=2, dim=1)
    B_norms = torch.norm(B, p=2, dim=1)
    cosines = inner_products/(A_norms*B_norms)
    if contains_nan(cosines):
        print('cosines contains nans in activation angles:')
    if torch.sum(A_norms == 0) > 0:
        print('A_norms contains zeros')
    if torch.sum(B_norms == 0) > 0:
        print('B_norms contains zeros')
    cosines = torch.min(cosines, torch.ones_like(cosines))
    angles = torch.acos(cosines)
    return 180/np.pi*torch.mean(angles)


class NetworkError(Exception):
    pass


def list_to_str(list_arg, delim=' '):
    """
    Convert a list of numbers into a string.
    Args:
        list_arg: List of numbers.
        delim (optional): Delimiter between numbers.
    Returns:
        List converted to string.
    """

    ret = ''
    for i, e in enumerate(list_arg):
        if i > 0:
            ret += delim
        ret += str(e)
    return ret


def str_to_list(string, delim=',', type='float'):
    """
    Convert a str (that originated from a list) back
    to a list of floats.
    """

    if string[0] in ('[', '(') and string[-1] in (']', ')'):
        string = string[1:-1]
    if type == 'float':
        lst = [float(num) for num in string.split(delim)]
    elif type == 'int':
        lst = [int(num) for num in string.split(delim)]
    elif type == 'str':
        lst = [s.strip().replace('"', '').replace("'", "") for s in string.split(delim)]
    else:
        raise ValueError('type {} not recognized'.format(type))

    return lst


def setup_summary_dict(args):
    """
    Setup the summary dictionary that is written to the performance
    summary file (in the result folder).
    This method adds the keyword "summary" to `shared`.
    Args:
        config: Command-line arguments.
        shared: Miscellaneous data shared among training functions (summary dict
            will be added to this :class:`argparse.Namespace`).
        experiment: Type of experiment. See argument `experiment` of method
            :func:`probabilistic.prob_mnist.train_bbb.run`.
        mnet: Main network.
        hnet (optional): Hypernetwork.
    """

    summary = dict()

    summary_keys = [
                        'acc_train_last',
                        'acc_train_best',
                        'loss_train_last',
                        'loss_train_best',
                        'acc_test_last',
                        'acc_test_best',
                        'acc_val_last',
                        'acc_val_best',
                        'acc_test_val_best',
                        'acc_train_val_best',
                        'loss_test_val_best',
                        'loss_train_val_best',
                        'loss_val_best',
                        'epoch_best_loss',
                        'epoch_best_acc',
                        'loss_test_last',
                        'loss_test_best',
                        'rec_loss',
                        'rec_loss_last',
                        'rec_loss_best',
                        'rec_loss_first',
                        'rec_loss_init',
                        'rec_loss_var_av',
                        'avg_time_per_epoch',
                        'finished']

    for k in summary_keys:
        if k == 'finished':
            summary[k] = 0
        else:
            summary[k] = -1

    save_summary_dict(args, summary)

    return summary


def save_summary_dict(args, summary):
    """
    Write a text file in the result folder that gives a quick
    overview over the results achieved so far.
    Args:
        args (Namespace): command line inputs
        summary (dict): summary dictionary
    """

    summary_fn = 'performance_overview.txt'
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    with open(os.path.join(args.out_dir, summary_fn), 'w') as f:
        for k, v in summary.items():
            if isinstance(v, list):
                f.write('%s %s\n' % (k, list_to_str(v)))
            elif isinstance(v, float):
                f.write('%s %f\n' % (k, v))
            elif isinstance(v, (np.ndarray, pandas.DataFrame)):
                pass
            else:
                f.write('%s %d\n' % (k, v))


def get_av_reconstruction_loss(network):
    """
    Get the average reconstruction loss of the network across its layers
    for the current mini-batch.
    Args:
        network (networks.DTPNetwork): network
    Returns (torch.Tensor):
        Tensor containing a scalar of the average reconstruction loss
    """

    reconstruction_losses = np.array([])

    for layer in network.layers[1:]:
        reconstruction_losses = np.append(reconstruction_losses,
                                           layer.reconstruction_loss)

        reconstruction_losses = list(filter(lambda x: x != None, reconstruction_losses))

    return np.mean(reconstruction_losses[:-1])


def int_to_one_hot(class_labels, nb_classes, device, soft_target=1.):
    """
    Convert tensor containing a batch of class indexes (int) to a tensor
    containing one hot vectors.
    """

    one_hot_tensor = torch.zeros((class_labels.shape[0], nb_classes),
                                 device=device)
    for i in range(class_labels.shape[0]):
        one_hot_tensor[i, class_labels[i]] = soft_target

    return one_hot_tensor


def one_hot_to_int(one_hot_tensor):
    return torch.argmax(one_hot_tensor, 1)


def dict2csv(dct, file_path):
    with open(file_path, 'w') as f:
        for key in dct.keys():
            f.write("{}, {} \n".format(key, dct[key]))


def process_lr(lr_str):
    """
    Process the lr provided by argparse.
    Args:
        lr_str (str): a string containing either a single float indicating the
            learning rate, or a list of learning rates, one for each layer of
            the network.
    Returns: a float or a numpy array of learning rates
    """

    if ',' in lr_str:
        return np.array(str_to_list(lr_str, ','))
    else:
        return float(lr_str)


def process_hdim(hdim_str):
    if ',' in hdim_str:
        return str_to_list(hdim_str, ',', type='int')
    else:
        return int(hdim_str)


def process_hdim_fb(hdim_str):
    if ',' in hdim_str:
        return str_to_list(hdim_str, ',', type='int')
    else:
        return [int(hdim_str)]


def process_hidden_activation(string):
    if ',' in string:
        lst = str_to_list(string, ',', type='str')
        for elem in lst:
            assert elem in ['tanh', 'relu', 'linear', 'leakyrelu', 'sigmoid']
        return lst
    else:
        assert string in ['tanh', 'relu', 'linear', 'leakyrelu', 'sigmoid']
        return string


def check_gpu():
    try:
        name = torch.cuda.current_device()
        print("Using CUDA device {}.".format(torch.cuda.get_device_name(name)))
    except AssertionError:
        print("No CUDA device found.")


def contains_nan(tensor):
    nb_nans = tensor != tensor
    nb_infs = tensor == float('inf')
    if isinstance(nb_nans, bool):
        return nb_nans or nb_infs
    else:
        return torch.sum(nb_nans) > 0 or torch.sum(nb_infs) > 0


def logit(x):
    if torch.sum(x < 1e-12) > 0 or torch.sum(x > 1 - 1e-12) > 0:
        warnings.warn('Input to inverse sigmoid is out of'
                      'bound: x={}'.format(x))
    inverse_sigmoid = torch.log(x / (1 - x))
    if contains_nan(inverse_sigmoid):
        raise ValueError('inverse sigmoid function outputted a NaN')
    return torch.log(x / (1 - x))


def plot_loss(summary, logdir, logplot=False):
    plt.figure()
    plt.plot(summary['loss_train'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Train loss')
    if logplot:
        plt.yscale('log')
    plt.savefig(os.path.join(logdir, 'loss_train.svg'))
    plt.close()
    plt.figure()
    plt.plot(summary['loss_test'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Test loss')
    if logplot:
        plt.yscale('log')
    plt.savefig(os.path.join(logdir, 'loss_test.svg'))
    plt.close()


def make_plot_output_space(args, net, i, loss_function,
                           targets, inputs, steps=20):
    """
    Make a plot of how the output activations would change if the update 
    for the parameters of layer(s) i is applied with a varying stepsize from 
    zero to one. 
    Args:
        args: command line arguments
        net: network
        i: layer index. If None, all layers are updated
        loss_function: loss function
        targets: true labels for the current batch
        inputs: batch with inputs for the network
        steps: amount of interpolation steps
    Returns: Saves a plot and the sequence of activations
    """

    if args.output_space_plot_bp:
        args.network_type = 'BP'

    inputs = inputs.flatten(1, -1)
    inputs = inputs[0:1, :]
    targets = targets[0:1, :]

    if i is None:
        parameters = net.get_forward_parameter_list()

    else:
        parameters = net.layers[i].get_forward_parameter_list()

    alpha = 1e-5
    sgd_optimizer = torch.optim.SGD(parameters, lr=alpha)
    sgd_optimizer.zero_grad()
    predictions = net.forward(inputs)
    loss = loss_function(predictions, targets)

    if args.output_space_plot_bp:
        gradients = torch.autograd.grad(loss, parameters)
        for i, param in enumerate(parameters):
            param.grad = gradients[i].detach()
    else:
        net.backward(loss, args.target_stepsize, save_target=False,
                     norm_ratio=args.norm_ratio)

    output_start = net.forward(inputs)

    sgd_optimizer.step()
    output_next = net.forward(inputs)

    output_update = (output_next - output_start)[0, 0:2].detach().cpu().numpy()

    ax = plt.axes()
    plot_contours(output_start[0, 0:2], targets[0, 0:2], loss_function, ax)

    distance = np.linalg.norm(output_start.detach().cpu().numpy() -
                              targets.detach().cpu().numpy())
    x_low = targets[0, 0].detach().cpu().numpy() - 1.1 * distance
    x_high = targets[0, 0].detach().cpu().numpy() + 1.1 * distance
    y_low = targets[0, 1].detach().cpu().numpy() - 1.1 * distance
    y_high = targets[0, 1].detach().cpu().numpy() + 1.1 * distance

    plt.ylim(y_low, y_high)
    plt.xlim(x_low, x_high)

    output_arrow = distance / 2 / np.linalg.norm(output_update) * output_update
    output_arrow_start = output_start[0, 0:2].detach().cpu().numpy()

    ax.arrow(output_arrow_start[0], output_arrow_start[1],
              output_arrow[0], output_arrow[1],
              width=0.05,
              head_width=0.3
              )

    file_name = 'output_space_updates_fig_' + args.network_type + '.svg'
    plt.savefig(os.path.join(args.out_dir, file_name))
    plt.close()

    file_name = 'output_arrow_' + args.network_type + '.npy'
    np.save(os.path.join(args.out_dir, file_name),
            output_arrow)
    file_name = 'output_arrow_start_' + args.network_type + '.npy'
    np.save(os.path.join(args.out_dir, file_name),
            output_arrow_start)
    file_name = 'output_space_label_' + args.network_type + '.npy'
    np.save(os.path.join(args.out_dir, file_name),
            targets[0, 0:2].detach().cpu().numpy())


def plot_contours(y, label, loss_function, ax, fontsize=26):
    """
    Make a 2D contour plot of loss_function(y, targets)
    """

    gridpoints = 100

    distance = np.linalg.norm(y.detach().cpu().numpy() -
                              label.detach().cpu().numpy())
    y1 = np.linspace(label[0].detach().cpu().numpy() - 1.1*distance,
                     label[0].detach().cpu().numpy() + 1.1*distance,
                     num=gridpoints)
    y2 = np.linspace(label[1].detach().cpu().numpy() - 1.1*distance,
                     label[1].detach().cpu().numpy() + 1.1*distance,
                     num=gridpoints)

    Y1, Y2 = np.meshgrid(y1, y2)

    L = np.zeros(Y1.shape)
    for i in range(gridpoints):
        for j in range(gridpoints):
            y_sample = torch.Tensor([Y1[i,j], Y2[i, j]])
            L[i,j] = loss_function(y_sample, label).item()

    levels = np.linspace(1.01*L.min(), L.max(), num=9)

    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    CS = ax.contour(Y1, Y2, L, levels=levels)


def make_plot_output_space_bp(args, net, i, loss_function,
                           targets, inputs, steps=20):
    """
    Make a plot of how the output activations would change if the update
    for the parameters of layer(s) i is applied with a varying stepsize from
    zero to one.
    Args:
        args: command line arguments
        net: network
        i: layer index. If None, all layers are updated
        loss_function: loss function
        targets: true labels for the current batch
        inputs: batch with inputs for the network
        steps: amount of interpolation steps
    Returns: Saves a plot and the sequence of activations
    """

    if i is None:
        parameters = net.parameters()

    else:
        parameters = net.layers[i].parameters()

    alpha = 1e-5
    sgd_optimizer = torch.optim.SGD(parameters, lr=alpha)
    sgd_optimizer.zero_grad()

    inputs = inputs.flatten(1, -1)
    inputs = inputs[0:1, :]
    targets = targets[0:1, :]

    predictions = net(inputs)
    loss = loss_function(predictions, targets)
    loss.backward()

    output_start = net(inputs)
    sgd_optimizer.step()
    output_next = net(inputs)

    output_update = (output_next - output_start)[0, 0:2].detach().cpu().numpy()

    ax = plt.axes()
    plot_contours(output_start[0, 0:2], targets[0, 0:2], loss_function, ax)

    distance = np.linalg.norm(output_start.detach().cpu().numpy() -
                              targets.detach().cpu().numpy())
    x_low = targets[0,0].detach().cpu().numpy() - 1.1 * distance
    x_high = targets[0, 0].detach().cpu().numpy() + 1.1 * distance
    y_low = targets[0, 1].detach().cpu().numpy() - 1.1 * distance
    y_high = targets[0, 1].detach().cpu().numpy() + 1.1 * distance

    plt.ylim(y_low, y_high)
    plt.xlim(x_low, x_high)

    output_arrow = distance / 2 / np.linalg.norm(output_update) * output_update
    output_arrow_start = output_start[0, 0:2].detach().cpu().numpy()

    ax.arrow(output_arrow_start[0], output_arrow_start[1],
              output_arrow[0], output_arrow[1],
              width=0.05,
              head_width=0.3
              )

    file_name = 'output_space_updates_fig_' + args.network_type + '.svg'
    plt.savefig(os.path.join(args.out_dir, file_name))
    plt.close()
    file_name = 'output_arrow_' + args.network_type + '.npy'
    np.save(os.path.join(args.out_dir, file_name),
            output_arrow)
    file_name = 'output_arrow_start_' + args.network_type + '.npy'
    np.save(os.path.join(args.out_dir, file_name),
            output_arrow_start)
    file_name = 'output_space_label_' + args.network_type + '.npy'
    np.save(os.path.join(args.out_dir, file_name),
            targets[0, 0:2].detach().cpu().numpy())


def nullspace(A, tol=1e-12):
    U, S, V = torch.svd(A, some=False)
    if S.min() >= tol:
        null_start = len(S)
    else:
        null_start = int(len(S) - torch.sum(S<tol))

    V_null = V[:, null_start:]
    return V_null


def nullspace_relative_norm(A, x, tol=1e-12):
    """
    Compute the ratio between the norm
    of components of x that are in the nullspace of A
    and the norm of x
    """

    if len(x.shape) == 1:
        x = x.unsqueeze(1)
    A_null = nullspace(A, tol=tol)
    x_null_coordinates = A_null.t().mm(x)
    ratio = x_null_coordinates.norm()/x.norm()
    return ratio


class time_counter():
    def __init__(self, task=" "):
        self.task = task
        self.t_start = 0.
        self.t_end = 0.

    def start(self):
        self.t_start = time.time()
        return self

    def end(self):
        self.t_end = time.time()
        print(self.task + " took %.2f ms to complete.\n" %((self.t_end - self.t_start)*1000) )
        return self

if __name__ == '__main__':
    pass


def dist(v1, v2, axis=None):
    """ Utility function to compute the Euclidean distance between two vectors.
    If only 1D vectors, an scalar is returned. If a 2D or 3D matrix is feed,
    the first dimension is interpreted as time and vector/matrix distance
    is computed along it."""

    if axis is None:
        if len(v1.shape) == 1: axis = 0        # normal 1D vectors
        elif len(v1.shape) == 2: axis = 1      # time x vector matrix, calculate along time
        elif len(v1.shape) == 3: axis = (1, 2)  # time x batch x vector matrix, calculate along time

    if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
        d = torch.norm(v1-v2, dim=axis, p=2).detach()
    elif isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
        d = np.linalg.norm(v1-v2, axis=axis)
    else:
        raise ValueError('Invalid types {},{} for dist function'.format(
            type(v1), type(v2)))

    return d


def split_in_layers(network, layers_concat):
    """
    Split a Tensor containing the concatenated layer activations for a
    minibatch into a list containing the activations of layer ``i`` at
    index ``i``.
    Args:
        network: network object
        layers_concat (torch.Tensor): a tensor of dimension
        :math:`B \times \sum_{l=1}^L n_l` containing the concatenated
            layer activations.

    Returns (list): A list containing values of ``layers_concat`` corresponding
    to the activations of layer ``i`` at index ``i``

    """
    layer_output_dims = [l.weights.shape[0] for l in network.layers]
    start_output_limits = [sum(layer_output_dims[:i]) for i in
                           range(len(network.layers))]
    end_output_limits = [sum(layer_output_dims[:i + 1]) for i in
                         range(len(network.layers))]
    if len(layers_concat.shape) == 1:
        # to avoid errors when batch_size==1
        layers_concat = layers_concat.unsqueeze(0)
    return [layers_concat[:, start_output_limits[i]:end_output_limits[i]]
                             for i in range(len(network.layers))]


def split_and_reshape_vectorized_parameters(network, vectorized_parameters):
    """
    Split a 1D Tensor containing values corresponding to
    the concatenated vectorized weights and biases
    (e.g. their updates) into a list containing the reshaped values, i.e. a
    separate entry in the list for each weight matrix or use_bias.
    Args:
        network: the network
        vectorized_parameters (torch.Tensor): values shaped in accordance with
            :math:`\bar{W}`, which is defined as
            ..math::
                \bar{W} &= [vec(W_1)^T \mathbf{b}_1^T ... vec(W_L)^T \mathbf{b}_L^T]^T
    Returns (list): A list containing the values of ``vectorized_parameters``
        reshaped and reordered, i.e. a separate entry in the list for the values
        corresponding to each weight matrix or use_bias.

    """
    layer_output_dims = [l.weights.shape[0] for l in network.layers]
    layer_input_dims = [l.weights.shape[1] for l in network.layers]
    start_indices = []
    end_indices = []
    running_index = 0
    for i in range(network.depth):
        start_indices.append(running_index)
        running_index += layer_input_dims[i]*layer_output_dims[i]
        end_indices.append(running_index)
        if network.use_bias:
            start_indices.append(running_index)
            running_index += layer_output_dims[i]
            end_indices.append(running_index)

    parameter_list = network.get_forward_parameter_list()
    splitted_parameters = []
    for i in range(len(start_indices)):
        splitted_parameters.append(
            vectorized_parameters[start_indices[i]: end_indices[i]].reshape_as(parameter_list[i])
        )

    return splitted_parameters


def vectorize_tensor_list(tensor_list):
    """ Vectorize all tensors in tensor_list and concatenate them in
    one single vector."""

    return torch.cat([t.view(-1).detach() for t in tensor_list])


def bool_to_indices(bool_tensor):
    """ Convert an array of boolean indices to integer indices"""
    indices_int = []
    for i in range(len(bool_tensor)):
        if bool_tensor[i]:
            indices_int.append(i)
    return indices_int


def fix_grad_norm_(parameters, fixed_norm, norm_type):
    """
    Inspired by implementation of clip_grad_norm_ in PyTorch

    Rescale the gradients such that the total norm of the gradients is
    equal to fixed_norm.
    Args:
        parameters (Iterable[torch.Tensor] or torch.Tensor): Iterable
            containing all the parameters that are optimized by the
            network
        fixed_norm (float or int): positive float indicating the total norm
            to which we want to rescale the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
    Returns:
        The total norm of the parameters before rescaling. This method
        changes the gradients of the parameters in-place.
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = [p for p in parameters if p.grad is not None]
    assert fixed_norm > 0
    fixed_norm = float(fixed_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                norm_type).to(device) for p in parameters]), norm_type)
    clip_coef = fixed_norm / (total_norm + 1e-6)
    for p in parameters:
        p.grad.detach().mul_(clip_coef.to(p.grad.device))
    return total_norm


class FastMNIST(MNIST):

    def __init__(self, root, device, double_precision, train=True, **kwargs ):
        super().__init__(root, train, **kwargs)

        self.data = self.data.unsqueeze(1).float().div(255)
        self.data = self.data.sub_(0.1307).div_(0.3081)

        if double_precision:
            self.data, self.targets = self.data.double().to(device), self.targets.to(device)
        else:
            self.data, self.targets = self.data.to(device), self.targets.to(device)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.data[index], self.targets[index]

        return img, target


class FastFashionMNIST(FashionMNIST):
    def __init__(self, root, device, double_precision, train=True, **kwargs ):
        super().__init__(root, train, **kwargs)

        self.data = self.data.unsqueeze(1).float().div(255)
        if double_precision:
            self.data, self.targets = self.data.double().to(device), self.targets.to(device)
        else:
            self.data, self.targets = self.data.to(device), self.targets.to(device)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.data[index], self.targets[index]

        return img, target