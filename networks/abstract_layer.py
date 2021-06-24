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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
import utils.utils as utils
from abc import ABC, abstractmethod


class AbstractLayer(nn.Module, ABC):

    def __init__(self, in_features, out_features, bias=True,
                 forward_requires_grad=False, forward_activation='tanh',
                 initialization='orthogonal'):
        nn.Module.__init__(self)

        self._weights = nn.Parameter(torch.Tensor(out_features, in_features),
                                     requires_grad=forward_requires_grad)

        if bias:
            self._bias = nn.Parameter(torch.Tensor(out_features),
                                      requires_grad=forward_requires_grad)
        else:
            self._bias = None

        if initialization == 'orthogonal':
            gain = np.sqrt(6. / (in_features + out_features))
            nn.init.orthogonal_(self._weights, gain=gain)
        elif initialization == 'xavier':
            nn.init.xavier_uniform_(self._weights)
        elif initialization == 'xavier_normal':
            nn.init.xavier_normal_(self._weights)
        elif initialization == 'teacher':
            nn.init.xavier_normal_(self._weights, gain=3.)
        else:
            raise ValueError('Provided weight initialization "{}" is not '
                             'supported.'.format(initialization))

        if bias:
            nn.init.constant_(self._bias, 0)

        self._activations = None
        self._linearactivations = None
        self._forward_activation = forward_activation
        self._use_bias = bias

    @property
    def weights(self):
        """Getter for read-only attribute :attr:`weights`."""
        return self._weights

    @property
    def bias(self):
        """Getter for read-only attribute :attr:`bias`."""
        return self._bias

    @property
    def activations(self):
        """Getter for read-only attribute :attr:`activations` """
        return self._activations

    @activations.setter
    def activations(self, value):
        """ Setter for the attribute activations"""
        self._activations = value

    @property
    def linearactivations(self):
        """Getter for read-only attribute :attr:`linearactivations` """
        return self._linearactivations

    @linearactivations.setter
    def linearactivations(self, value):
        """Setter for the attribute :attr:`linearactivations` """
        self._linearactivations = value

    @property
    def forward_activation(self):
        """ Getter for read-only attribute forward_activation"""
        return self._forward_activation

    @property
    def use_bias(self):
        """ Getter for read-only attribute :attr:`use_bias`"""
        return self._use_bias


    def get_forward_parameter_list(self):
        """
        Return forward weights and forward bias if applicable
        """

        parameterlist = []
        parameterlist.append(self.weights)
        if self.bias is not None:
            parameterlist.append(self.bias)
        return parameterlist


    def forward_activationfunction(self, x):
        """
        Element-wise forward activation function
        """

        if self.forward_activation == 'tanh':
            return torch.tanh(x)
        elif self.forward_activation == 'relu':
            return F.relu(x)
        elif self.forward_activation == 'linear':
            return 1.*x
        elif self.forward_activation == 'leakyrelu':
            return F.leaky_relu(x, 0.2)
        elif self.forward_activation == 'sigmoid':
            return torch.sigmoid(x)
        else:
            raise ValueError('The provided forward activation {} is not '
                             'supported'.format(self.forward_activation))


    def compute_vectorized_jacobian(self, a):
        """
        Compute the vectorized Jacobian of the forward activation function,
        evaluated at a. The vectorized Jacobian is the vector with the diagonal
        elements of the real Jacobian, as it is a diagonal matrix for element-
        wise functions. As a is a minibatch, the output will also be a
        mini-batch of vectorized Jacobians (thus a matrix).
        Args:
            a (torch.Tensor): linear activations
        """

        if self.forward_activation == 'tanh':
            return 1. - torch.tanh(a)**2
        elif self.forward_activation == 'relu':
            J = torch.ones_like(a)
            J[a < 0.] = 0.
            return J
        elif self.forward_activation == 'leakyrelu':
            J = torch.ones_like(a)
            J[a < 0.] = 0.2
            return J
        elif self.forward_activation == 'linear':
            return torch.ones_like(a)
        elif self.forward_activation == 'sigmoid':
            s = torch.sigmoid(a)
            return s * (1 - s)
        else:
            raise ValueError('The provided forward activation {} is not '
                             'supported'.format(self.forward_activation))


    def requires_grad(self):
        """
        Set require_grad attribute of the activations of this layer to
        True, such that the gradient will be saved in the activation tensor.
        """

        self._activations.requires_grad = True


    def forward(self, x):
        """
        Compute the output activation of the layer.
        This method applies first a linear mapping with the
        parameters ``weights`` and ``bias``, after which it applies the
        forward activation function.
        Args:
            x: A mini-batch of size B x in_features with input activations from
            the previous layer or input.
        Returns:
            The mini-batch of output activations of the layer.
        """

        a = x.mm(self.weights.t())
        if self.bias is not None:
            a += self.bias.unsqueeze(0).expand_as(a)
        self.linearactivations = a

        self.activations = self.forward_activationfunction(a)
        return self.activations


    def dummy_forward(self, x):
        """
        Same as the forward method, besides that the activations and
        linear activations are not saved in self.
        """

        a = x.mm(self.weights.t())
        if self.bias is not None:
            a += self.bias.unsqueeze(0).expand_as(a)
        h = self.forward_activationfunction(a)
        return h

    def dummy_forward_linear(self, x):
        """
        Propagate the input of the layer forward to the linear activation
        of the current layer (so no nonlinearity applied), without saving the
        linear activations.
        """

        a = x.mm(self.weights.t())
        if self.bias is not None:
            a += self.bias.unsqueeze(0).expand_as(a)

        return a

    @abstractmethod
    def compute_forward_gradients(self, h_target, h_previous):
        """
        Compute the updates for the forward parameters, based on
        a target activation and the activation of the previous layer.
        """

        pass

    def compute_bp_update(self, loss, retain_graph=False):
        """
        Compute the error backpropagation update for the forward
        parameters of this layer, based on the given loss.
        Args:
            loss (nn.Module): network loss
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.
        """

        if self.bias is not None:
            grads = torch.autograd.grad(loss, [self.weights, self.bias],
                                        retain_graph=retain_graph)
        else:
            grads = torch.autograd.grad(loss, self.weights,
                                        retain_graph=retain_graph)

        return grads

    def compute_bp_activation_updates(self, loss, retain_graph=False,
                                      linear=False):
        """
        Compute the error backpropagation teaching signal for the
        activations of this layer, based on the given loss.
        Args:
            loss (nn.Module): network loss
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.
            linear (bool): Flag indicating whether the GN update for the
                linear activations should be computed instead of for the
                nonlinear activations.
        Returns (torch.Tensor): A tensor containing the BP updates for the layer
            activations for the current mini-batch.
        """

        if linear:
            activations = self.linearactivations
        else:
            activations = self.activations
        grads = torch.autograd.grad(loss, activations,
                                    retain_graph=retain_graph)[0].detach()
        return grads


    def compute_nullspace_relative_norm(self, output_activation, retain_graph=False):
        """
        Compute the norm of the components of weights.grad that are in the nullspace
        of the jacobian of the output with respect to weights, relative to the norm of
        weights.grad.
        """

        J = utils.compute_jacobian(self.weights, output_activation,
                                   structured_tensor=False,
                                   retain_graph=retain_graph)
        weights_update_flat = self.weights.grad.view(-1)
        relative_norm = utils.nullspace_relative_norm(J, weights_update_flat)
        return relative_norm


    def save_logs(self, writer, step, name):
        """
        Save logs and plots of this layer on tensorboardX
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
            name (str): The name of the layer
        """

        forward_weights_norm = torch.norm(self.weights)
        writer.add_scalar(tag='{}/forward_weights_norm'.format(name),
                          scalar_value=forward_weights_norm,
                          global_step=step)
        if self.weights.grad is not None:
            forward_weights_gradients_norm = torch.norm(self.weights.grad)
            writer.add_scalar(tag='{}/forward_weights_gradients_norm'.format(name),
                              scalar_value=forward_weights_gradients_norm,
                              global_step=step)
        if self.bias is not None:
            forward_bias_norm = torch.norm(self.bias)

            writer.add_scalar(tag='{}/forward_bias_norm'.format(name),
                              scalar_value=forward_bias_norm,
                              global_step=step)
            if self.bias.grad is not None:
                forward_bias_gradients_norm = torch.norm(self.bias.grad)
                writer.add_scalar(tag='{}/forward_bias_gradients_norm'.format(name),
                                  scalar_value=forward_bias_gradients_norm,
                                  global_step=step)


    def get_forward_parameters(self):
        """
        Return a list containing the forward parameters.
        """

        if self.bias is not None:
            return [self.weights, self.bias]
        else:
            return [self.weights]

    def get_forward_gradients(self):
        """
        Return a tuple containing the gradients of the forward
        parameters.
        """

        if self.bias is not None:
            return (self.weights.grad, self.bias.grad)
        else:
            return (self.weights.grad, )