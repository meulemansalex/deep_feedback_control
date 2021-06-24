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
import numpy as np
from networks.abstract_network import AbstractNetwork
from target_propagation.dtp_layers import DTPLayer
from target_propagation.dtpdrl_layers import DTPDRLLayer
from tensorboardX import SummaryWriter
import pandas as pd


class DTPNetwork(AbstractNetwork):
    """
    A multilayer perceptron (MLP) network that will be trained by the
    difference target propagation (DTP) method.
    Attributes:
        layers (nn.ModuleList): a ModuleList with the layer objects of the MLP
        depth: the depth of the network (# hidden layers + 1)
        input (torch.Tensor): the input minibatch of the current training
                iteration. We need
                to save this tensor for computing the weight updates for the
                first hidden layer
        sigma: standard deviation of the gaussian that corrupts layer
                activations for computing the reconstruction losses.
        update_idx (None or int): the layer index of which the layer parameters
            are updated for the current mini-batch, when working in a randomized
            setting. If the randomized setting is not used, it is equal to None.
    Args:
        n_in: input dimension (flattened input assumed)
        n_hidden: list with hidden layer dimensions
        n_out: output dimension
        activation: activation function indicator for the hidden layers
        output_activation: activation function indicator for the output layer
        bias: boolean indicating whether the network uses biases or not
        sigma: standard deviation of the gaussian that corrupts layer
                activations for computing the reconstruction losses.
        forward_requires_grad (bool): Flag indicating whether the forward
            parameters require gradients that can be computed with autograd.
            This might be needed when comparing the DTP updates with BP updates
            and GN updates.
        initialization (str): the initialization method used for the forward
                and feedback matrices of the layers
    """

    def __init__(self, n_in, n_hidden, n_out, activation='relu',
                 output_activation='linear', bias=True, sigma=0.36,
                 forward_requires_grad=False,
                 initialization='orthogonal',
                 fb_activation='relu',
                 save_df=False,
                 clip_grad_norm=-1):
        super().__init__(n_in=n_in,
                         n_hidden=n_hidden,
                         n_out=n_out,
                         activation=activation,
                         output_activation=output_activation,
                         bias=bias,
                         forward_requires_grad=forward_requires_grad,
                         initialization=initialization,
                         save_df=save_df,
                         clip_grad_norm=clip_grad_norm)

        self._layers = self.set_layers(n_in, n_hidden, n_out, activation,
                                       output_activation, bias,
                                       forward_requires_grad,
                                       initialization,
                                       fb_activation)
        self._sigma = sigma
        self._update_idx = None

        if save_df:
            self.gn_angles = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.gnt_angles = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.ndi_angles = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.bp_activation_angles = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.gn_activation_angles = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.reconstruction_loss_init = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.reconstruction_loss = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])


    def set_layers(self, n_in, n_hidden, n_out, activation, output_activation,
                   bias, forward_requires_grad, initialization,
                   fb_activation=None):
        """
        Create the layers of the network and output them as a ModuleList.
        Args:
            n_in: input dimension (flattened input assumed)
            n_hidden: list with hidden layer dimensions
            n_out: output dimension
            activation: activation function indicator for the hidden layers
            output_activation: activation function indicator for the output
                layer
            bias: boolean indicating whether the network uses biases or not
            forward_requires_grad (bool): Flag indicating whether the forward
            parameters require gradients that can be computed with autograd.
            This might be needed when comparing the DTP updates with BP updates
            and GN updates.
            initialization (str): the initialization method used for the forward
                and feedback matrices of the layers
            fb_activation (str): activation function indicator for the feedback
                path of the hidden layers
        """

        n_all = [n_in] + n_hidden + [n_out]
        layers = nn.ModuleList()
        if isinstance(activation, str):
            activation = [activation]*len(n_hidden)

        for i in range(1, len(n_all) - 1):
            layers.append(
                DTPLayer(n_all[i - 1], n_all[i], bias=bias,
                         forward_activation=activation[i-1],
                         feedback_activation=fb_activation,
                         forward_requires_grad=forward_requires_grad,
                         initialization=initialization
                         ))
        layers.append(DTPLayer(n_all[-2], n_all[-1], bias=bias,
                               forward_activation=output_activation,
                               feedback_activation=fb_activation,
                               forward_requires_grad=forward_requires_grad,
                               initialization=initialization))
        return layers

    @property
    def sigma(self):
        """ Getter for read-only attribute sigma"""
        return self._sigma

    @property
    def update_idx(self):
        """ Getter for attribute update_idx"""
        return self._update_idx

    @update_idx.setter
    def update_idx(self, value):
        """Setter for attribute update_idx"""
        self._update_idx = value

    def propagate_backward(self, h_target, i):
        """
        Propagate the output target backwards to layer i in a DTP-like fashion.
        Args:
            h_target (torch.Tensor): the output target
            i: the layer index to which the target must be propagated
        Returns: the target for layer i
        """

        for k in range(self.depth-1, i, -1):
            h_current = self.layers[k].activations
            h_previous = self.layers[k-1].activations
            h_target = self.layers[k].backward(h_target, h_previous, h_current)
        return h_target


    def backward_random(self, loss, target_lr, i, save_target=False,
                        norm_ratio=1.):
        """
        Propagate the output target backwards through the network until
        layer i. Based on this target, compute the gradient of the forward
        weights and use_bias of layer i and save them in the parameter tensors.
        Args:
            loss (nn.Module): output loss of the network
            target_lr: the learning rate for computing the output target based
                on the gradient of the loss w.r.t. the output layer
            i: layer index to which the target needs to be propagated and the
                gradients need to be computed
            save_target (bool): flag indicating whether the target should be
                saved in the layer object for later use.
            norm_ratio (float): will only be used in children of DTPLayer for
                the minimal_norm update
        """

        self.update_idx = i

        h_target = self.compute_output_target(loss, target_lr)

        h_target = self.propagate_backward(h_target, i)

        if save_target:
            self.layers[i].target = h_target

        if i == 0:
            self.layers[i].compute_forward_gradients(h_target, self.input,
                                                     norm_ratio=norm_ratio)
        else:
            self.layers[i].compute_forward_gradients(h_target,
                                                 self.layers[i-1].activations,
                                                     norm_ratio=norm_ratio)


    def backward_all(self, output_target, save_target=False, norm_ratio=1.):
        """
        Propagate the output_target backwards through all the layers. Based
        on these local targets, compute the gradient of the forward weights and
        biases of all layers.
        Args:
            output_target (torch.Tensor): a mini-batch of targets for the
                output layer.
            save_target (bool): flag indicating whether the target should be
                saved in the layer object for later use.
            norm_ratio (float): will only be used in children of DTPLayer for
                the minimal_norm update
        """

        h_target = output_target

        if save_target:
            self.layers[-1].target = h_target
        for i in range(self.depth-1, 0, -1):
            h_current = self.layers[i].activations
            h_previous = self.layers[i-1].activations
            self.layers[i].compute_forward_gradients(h_target, h_previous,
                                                     norm_ratio=norm_ratio)
            h_target = self.layers[i].backward(h_target, h_previous, h_current)
            if save_target:
                self.layers[i-1].target = h_target

        self.layers[0].compute_forward_gradients(h_target, self.input,
                                                 norm_ratio=norm_ratio)


    def backward(self, loss, target_lr, save_target=False, norm_ratio=1.):
        """
        Compute and propagate the output_target backwards through all the
        layers. Based on these local targets, compute the gradient of the
        forward weights and biases of all layers.
        Args:
            loss (torch.Tensor): output loss of the network
            target_lr: the learning rate for computing the output target based
                on the gradient of the loss w.r.t. the output layer
            save_target (bool): flag indicating whether the target should be
                saved in the layer object for later use.
            norm_ratio (float): will only be used in children of DTPLayer for
                the minimal_norm update
        """

        output_target = self.compute_output_target(loss, target_lr)
        self.backward_all(output_target, save_target, norm_ratio=norm_ratio)


    def compute_feedback_gradients(self):
        """
        Compute the local reconstruction loss for each layer and compute
        the gradient of those losses with respect to
        the feedback weights and biases. The gradients are saved in the
        feedback parameter tensors.
        """

        for i in range(1, self.depth):
            h_corrupted = self.layers[i-1].activations + \
                    self.sigma * torch.randn_like(self.layers[i-1].activations)
            self.layers[i].compute_feedback_gradients(h_corrupted, self.sigma)


    def get_reduced_forward_parameter_list(self):
        """
        Returns: a list with all the forward parameters of the network, except
        from the ones of the output layer.
        """

        parameterlist = []
        for layer in self.layers[:-1]:
            parameterlist.append(layer.weights)
            if layer.use_bias is not None:
                parameterlist.append(layer.use_bias)
        return parameterlist


    def get_forward_parameters_last_two_layers(self):
        parameterlist = []
        for layer in self.layers[-2:]:
            parameterlist.append(layer.weights)
            if layer.use_bias is not None:
                parameterlist.append(layer.use_bias)
        return parameterlist


    def get_forward_parameters_last_three_layers(self):
        parameterlist = []
        for layer in self.layers[-3:]:
            parameterlist.append(layer.weights)
            if layer.use_bias is not None:
                parameterlist.append(layer.use_bias)
        return parameterlist


    def get_forward_parameters_last_four_layers(self):
        parameterlist = []
        for layer in self.layers[-4:]:
            parameterlist.append(layer.weights)
            if layer.use_bias is not None:
                parameterlist.append(layer.use_bias)
        return parameterlist


    def get_forward_parameter_list_first_layer(self):
        """
        Returns: a list with only the forward parameters of the first layer.
        """

        parameterlist = []
        parameterlist.append(self.layers[0].weights)
        if self.layers[0].use_bias is not None:
            parameterlist.append(self.layers[0].use_bias)
        return parameterlist


    def get_feedback_parameter_list(self):
        """
        Returns (list): a list with all the feedback parameters (weights and
            biases) of the network. Note that the first hidden layer does not
            need feedback parameters, so they are not put in the list.
        """

        parameterlist = []
        for layer in self.layers[1:]:
            parameterlist.append(layer.feedbackweights)
            if layer.feedbackbias is not None:
                parameterlist.append(layer.feedbackbias)
        return parameterlist


    def save_logs(self, writer, step):
        """
        Save logs and plots for tensorboardX.
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
        """

        for i in range(len(self.layers)):
            name = 'layer {}'.format(i+1)
            self.layers[i].save_logs(writer, step, name,
                                     no_gradient=i==0,
                                     no_fb_param= False)  # added for TPDI


    def save_feedback_batch_logs(self, writer, step, init=False,
                                 retain_graph=False, save_tensorboard=True,
                                 save_dataframe=True, save_statistics=False,
                                 damping=0):
        """
        Save the logs for the current minibatch on tensorboardX.
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
            init (bool): flag indicating that the training is in the
                initialization phase (only training the feedback weights).
        """

        for i in range(len(self.layers)):
            name = 'layer {}'.format(i+1)
            self.layers[i].save_feedback_batch_logs(writer, step, name,
                                     no_gradient=i == 0, init=init)


    def get_av_reconstruction_loss(self):
        """
        Get the average reconstruction loss of the network across its layers
        for the current mini-batch.
        Args:
            network (networks.DTPNetwork): network
        Returns (torch.Tensor):
            Tensor containing a scalar of the average reconstruction loss
        """

        reconstruction_losses = np.array([])

        for layer in self.layers[1:]:
            reconstruction_losses = np.append(reconstruction_losses,
                                              layer.reconstruction_loss)
        if np.any([reconstruction_losses[i] is not None for i in range(len(reconstruction_losses))]):
            return np.mean(reconstruction_losses)
        else:
            return 0.


class LeeDTPNetwork(nn.Module):
    """ Class for the DTP network used in Lee2015 to classify MNIST digits. In
    this network, the target for the last hidden layer is computed by error
    backpropagation instead of DTP. """

    def __init__(self, n_in, n_hidden, n_out, activation='tanh',
                 output_activation='linear', bias=True, sigma=0.36,
                 initialization='orthogonal',
                 forward_requires_grad=False):
        """
        See arguments of __init__ of class DTP Network
        Attributes:
            dtpnetwork (DTPNetwork): a DTP Network of all the layers except
                from the output
                layer. These layers will be trained by the DTP method.
            linearlayer (nn.Linear): the output linear layer. On this layer, the
                CrossEntropyLoss will be applied during training.
            hiddengradient: the gradient of the loss with respect to the
                activation of the last hidden layer of the network.
            depth (int): depth of the network (number of hidden layers + 1)
            initialization (str): the initialization method used for the forward
                and feedback matrices of the layers
        """

        nn.Module.__init__(self)

        self._dtpnetwork = DTPNetwork(n_in, n_hidden[:-1], n_hidden[-1],
                                      activation=activation,
                                      output_activation=activation,
                                      bias=bias, sigma=sigma,
                                      initialization=initialization,
                                      forward_requires_grad=
                                      forward_requires_grad)

        self._linearlayer = nn.Linear(n_hidden[-1], n_out, bias=bias)
        if initialization == 'orthogonal':
            gain = np.sqrt(6./(n_hidden[-1] + n_out))
            nn.init.orthogonal_(self._linearlayer.weight, gain=gain)
        elif initialization == 'xavier':
            nn.init.xavier_uniform_(self._linearlayer.weight)
        else:
            raise ValueError('Given initialization "{}" is not supported.'\
                             .format(initialization))
        if bias:
            nn.init.constant_(self._linearlayer.bias, 0)
        self._depth = len(n_hidden) + 1

        if output_activation != 'linear':
            raise ValueError('{} is not supported as an output '
                             'activation'.format(output_activation))

        self._update_idx = None
        self._forward_requires_grad = forward_requires_grad

    @property
    def dtpnetwork(self):
        """ Getter for read-only attribute dtpnetwork"""
        return self._dtpnetwork

    @property
    def linearlayer(self):
        """ Getter for read-only attribute linearlayer"""
        return self._linearlayer

    @property
    def depth(self):
        """Getter for read-only attribute :attr:`depth`."""
        return self._depth

    @property
    def update_idx(self):
        """ Getter for attribute update_idx"""
        return self._update_idx

    @update_idx.setter
    def update_idx(self, value):
        """Setter for attribute update_idx"""
        self._update_idx = value

    @property
    def forward_requires_grad(self):
        """ Getter for read-only attribute forward_requires_grad"""
        return self._forward_requires_grad

    @property
    def layers(self):
        """Getter for attribute layers.
        Warning: only the layers of the dtp network are returned, not the
        extra linear layer output layer"""
        return self.dtpnetwork.layers


    def forward(self, x):
        x = self.dtpnetwork.forward(x)
        if x.requires_grad == False:
            x.requires_grad = True
        x = self.linearlayer(x)
        return x


    def backward(self, loss, target_lr, save_target=False):
        """
        Compute the gradients of the output weights and use_bias, compute
        the target for the last hidden layer based on backprop, propagate target
        backwards and compute parameter updates for the rest of the DTP network.
        """

        gradients = torch.autograd.grad(loss, self.linearlayer.parameters(),
                                        retain_graph=True)
        for i, param in enumerate(self.linearlayer.parameters()):
            param.grad = gradients[i].detach()

        hidden_activations = self.dtpnetwork.layers[-1].activations
        hiddengradient = torch.autograd.grad(loss, hidden_activations,
                                             retain_graph=
                                             self.forward_requires_grad)
        hiddengradient = hiddengradient[0].detach()


        hidden_targets = hidden_activations - target_lr*hiddengradient
        self.dtpnetwork.backward_all(hidden_targets, save_target)


    def compute_feedback_gradients(self):
        """
        Compute the local reconstruction loss for each layer of the
        dtp network and compute the gradient of those losses with respect to
        the feedback weights and biases. The gradients are saved in the
        feedback parameter tensors.
        """

        self.dtpnetwork.compute_feedback_gradients()


    def get_forward_parameter_list(self):
        """
        Returns: a list with all the forward parameters (weights and biases) of
            the network.
        """

        parameterlist = self.dtpnetwork.get_forward_parameter_list()
        parameterlist.append(self.linearlayer.weight)
        if self.linearlayer.bias is not None:
            parameterlist.append(self.linearlayer.bias)
        return parameterlist


    def get_feedback_parameter_list(self):
        """
        Returns (list): a list with all the feedback parameters (weights and
            biases) of the network. Note that the first hidden layer does not
            need feedback parameters, so they are not put in the list.
        """

        return self.dtpnetwork.get_feedback_parameter_list()


    def get_reduced_forward_parameter_list(self):
        """
        Get the forward parameters of all the layers that will be trained by
        DTP, and not by BP (thus all the layer parameters except from the output
        layer and the last hidden layer.
        Returns: a list with all the parameters that will be trained by
            difference target propagtion
        """

        if self.dtpnetwork.layers[-1].use_bias is not None:
            return self.dtpnetwork.get_forward_parameter_list()[:-2]
        else:
            return self.dtpnetwork.get_forward_parameter_list()[:-1]


    def save_logs(self, writer, step):
        """
        Save logs and plots for tensorboardX.
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
        """

        self.dtpnetwork.save_logs(writer, step)

        output_weights = self.linearlayer.weight
        output_bias = self.linearlayer.bias

        name = 'layer {}'.format(self.dtpnetwork.depth + 1)

        forward_weights_norm = torch.norm(output_weights)
        forward_bias_norm = torch.norm(output_bias)

        forward_weights_gradients_norm = torch.norm(output_weights.grad)
        forward_bias_gradients_norm = torch.norm(output_bias.grad)

        writer.add_scalar(tag='{}/forward_weights_norm'.format(name),
                          scalar_value=forward_weights_norm,
                          global_step=step)
        writer.add_scalar(tag='{}/forward_bias_norm'.format(name),
                          scalar_value=forward_bias_norm,
                          global_step=step)
        writer.add_scalar(tag='{}/forward_weights_gradients_norm'.format(name),
                          scalar_value=forward_weights_gradients_norm,
                          global_step=step)
        writer.add_scalar(tag='{}/forward_bias_gradients_norm'.format(name),
                          scalar_value=forward_bias_gradients_norm,
                          global_step=step)


    def save_feedback_batch_logs(self, writer, step, init=False,
                                 retain_graph=False, save_tensorboard=True,
                                 save_dataframe=True, save_statistics=False,
                                 damping=0):
        """
        Save the logs for the current minibatch on tensorboardX.
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
        """

        self.dtpnetwork.save_feedback_batch_logs(writer, step)


    def save_bp_angles(self, writer, step, loss, retain_graph=False):
        """
        See DTPNetwork.save_bp_angles

        """

        self.dtpnetwork.save_bp_angles(writer, step, loss, retain_graph)


    def save_gn_angles(self, writer, step, output_activation, loss, damping,
                       retain_graph=False):
        """
        See DTPNetwork.save_gn_angles
        """

        self.dtpnetwork.save_gn_angles(writer, step, output_activation, loss,
                                       damping, retain_graph)


    def save_bp_activation_angle(self, writer, step, loss,
                                 retain_graph=False):
        """
        See DTPNetwork.save_bp_activation_angle.
        """

        self.dtpnetwork.save_bp_activation_angle(writer, step, loss,
                                 retain_graph)


    def save_gn_activation_angle(self, writer, step, output_activation, loss,
                                 damping, retain_graph=False):
        """
        See DTPNetwork.save_gn_activation_angle.
        """

        self.dtpnetwork.save_gn_activation_angle(writer, step,
                                                 output_activation, loss,
                                                 damping, retain_graph)


    def get_av_reconstruction_loss(self):
        """
        Get the average reconstruction loss of the network across its layers
        for the current mini-batch.
        Args:
            network (networks.DTPNetwork): network
        Returns (torch.Tensor):
            Tensor containing a scalar of the average reconstruction loss
        """
        return self.dtpnetwork.get_av_reconstruction_loss()



class DTPDRLNetwork(DTPNetwork):
    """
    A class for networks that contain DTPDRLLayers.
    """

    def set_layers(self, n_in, n_hidden, n_out, activation, output_activation,
                   bias, forward_requires_grad, initialization,
                   fb_activation):
        """
        Create the layers of the network and output them as a ModuleList.
        Args:
            n_in: input dimension (flattened input assumed)
            n_hidden: list with hidden layer dimensions
            n_out: output dimension
            activation: activation function indicator for the hidden layers
            output_activation: activation function indicator for the output
                layer
            bias: boolean indicating whether the network uses biases or not
            forward_requires_grad (bool): Flag indicating whether the forward
            parameters require gradients that can be computed with autograd.
            This might be needed when comparing the DTP updates with BP updates
            and GN updates.
            initialization (str): the initialization method used for the forward
                and feedback matrices of the layers
            fb_activation (str): activation function indicator for the feedback
                path of the hidden layers
        """
        n_all = [n_in] + n_hidden + [n_out]
        layers = nn.ModuleList()
        if isinstance(activation, str):
            activation = [activation]*len(n_hidden)

        for i in range(1, len(n_all) - 1):
            layers.append(
                DTPDRLLayer(n_all[i - 1], n_all[i], bias=bias,
                            forward_activation=activation[i-1],
                            feedback_activation=fb_activation,
                            forward_requires_grad=forward_requires_grad,
                            initialization=initialization
                            ))
        layers.append(DTPDRLLayer(n_all[-2], n_all[-1], bias=bias,
                                  forward_activation=output_activation,
                                  feedback_activation=fb_activation,
                                  forward_requires_grad=forward_requires_grad,
                                  initialization=initialization))
        return layers


    def compute_feedback_gradients(self, i):
        """
        Compute the difference reconstruction loss for layer i of the network
        and compute the gradient of this loss with respect to the feedback
        parameters. The gradients are saved in the .grad attribute of the
        feedback parameter tensors.
        """

        self.reconstruction_loss_index = i

        h_corrupted = self.layers[i-1].activations + \
            self.sigma * torch.randn_like(self.layers[i - 1].activations)
        output_corrupted = self.dummy_forward(h_corrupted, i-1)
        h_current_reconstructed = self.propagate_backward(output_corrupted, i)
        self.layers[i].compute_feedback_gradients(h_corrupted,
                                                  h_current_reconstructed,
                                                  self.layers[i-1].activations,
                                                  self.sigma)


    def dummy_forward(self, h, i):
        """
        Propagates the activations h of layer i forward to the output of the
        network, without saving activations and linear activations in the layer
        objects.
        Args:
            h (torch.Tensor): activations
            i (int): index of the layer of which h are the activations

        Returns: output of the network with h as activation for layer i
        """

        y = h

        for layer in self.layers[i+1:]:
            y = layer.dummy_forward(y)

        return y


    def get_av_reconstruction_loss(self):
        """ 
        Get the reconstruction loss of the network for the layer of which
        the feedback parameters were trained on the current mini-batch
        Returns (torch.Tensor):
            Tensor containing a scalar of the average reconstruction loss
        """
        reconstruction_loss = self.layers[self.reconstruction_loss_index].\
            reconstruction_loss
        return reconstruction_loss


class GNTNetwork(DTPDRLNetwork):
    """ 
    Network that computes exact GN targets for the nonlinear hidden layer
    activations and computes parameter updates using a gradient step on the
    local loss.
    """

    def __init__(self, n_in, n_hidden, n_out, activation='relu',
                 output_activation='linear', bias=True, sigma=0.36,
                 forward_requires_grad=False,
                 initialization='orthogonal',
                 fb_activation='relu',
                 plots=None,
                 damping=0.):
        super().__init__(n_in=n_in,
                         n_hidden=n_hidden,
                         n_out=n_out,
                         activation=activation,
                         output_activation=output_activation,
                         bias=bias,
                         sigma=sigma,
                         forward_requires_grad=forward_requires_grad,
                         initialization=initialization,
                         fb_activation=fb_activation,
                         plots=plots)
        self._damping = damping

    @property
    def damping(self):
        """ Getter for read only attr damping"""
        return self._damping


    def backward_random(self, loss, target_lr, i,
                        norm_ratio=1., save_target=False):

        self.update_idx = i
        output_activation = self.layers[-1].activations

        layer_update = self.layers[i].compute_gn_activation_updates(
                                      output_activation, loss,
                                      damping=self.damping,
                                      retain_graph=self.forward_requires_grad,
                                      linear=False).detach()

        h_target = self.layers[i].activations - layer_update

        if save_target:
            self.layers[i].target = h_target

        if i == 0:
            self.layers[i].compute_forward_gradients(h_target, self.input,
                                                     norm_ratio=norm_ratio)
        else:
            self.layers[i].compute_forward_gradients(h_target,
                                            self.layers[i-1].activations,
                                                     norm_ratio=norm_ratio)


    def backward(self, loss, target_lr, save_target=False, norm_ratio=1.):
        output_activation = self.layers[-1].activations

        for i in range(self.depth):
            if i == self.depth -1:
                retain_graph = self.forward_requires_grad
            else:
                retain_graph = True

            layer_update = self.layers[i].compute_gn_activation_updates(
                                      output_activation, loss,
                                      damping=self.damping,
                                      retain_graph=retain_graph,
                                      linear=False).detach()

            h_target = self.layers[i].activations - layer_update
            if save_target:
                self.layers[i].target = h_target

            if i == 0:
                self.layers[i].compute_forward_gradients(h_target, self.input,
                                                         norm_ratio=norm_ratio)
            else:
                self.layers[i].compute_forward_gradients(h_target,
                                                         self.layers[
                                                             i - 1].activations,
                                                         norm_ratio=norm_ratio)
    def compute_feedback_gradients(self):
        pass



