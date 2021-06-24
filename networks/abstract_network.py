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
from tensorboardX import SummaryWriter
import utils.utils as utils
import pandas as pd
from abc import ABC, abstractmethod


class AbstractNetwork(ABC, nn.Module):

    def __init__(self, n_in, n_hidden, n_out, activation='relu',
                 output_activation='linear', bias=True,
                 forward_requires_grad=False,
                 initialization='orthogonal',
                 save_df=False,
                 clip_grad_norm=-1):
        nn.Module.__init__(self)

        self._depth = len(n_hidden) + 1
        self._layers = None
        self._input = None
        self._forward_requires_grad = forward_requires_grad
        self._use_bias = bias
        self._save_df = save_df
        self._clip_grad_norm = clip_grad_norm
        if save_df:
            self.bp_angles = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.nullspace_relative_norm = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.gn_angles = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.gnt_angles = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.ndi_angles = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.ndi_angles_network = pd.DataFrame(columns=[0])
            self.jac_pinv_angles = pd.DataFrame(columns=[0])
            self.jac_transpose_angles = pd.DataFrame(columns=[0])
            self.jac_pinv_angles_init = pd.DataFrame(columns=[0])
            self.jac_transpose_angles_init = pd.DataFrame(columns=[0])
            self.gn_angles_network = pd.DataFrame(columns=[0])
            self.gnt_angles_network = pd.DataFrame(columns=[0])

    @abstractmethod
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
                path of the hidden layers. By default this is put to None, for
                the networks that don't need feedback activations.
        """
        pass

    @property
    def clip_grad_norm(self):
        """ Getter for read-only attribute :attr:`clip_grad_norm`."""
        return self._clip_grad_norm

    @property
    def depth(self):
        """Getter for read-only attribute :attr:`depth`."""
        return self._depth

    @property
    def use_bias(self):
        """Getter for read-only attribute :attr:`use_bias`."""
        return self._use_bias

    @property
    def layers(self):
        """Getter for read-only attribute :attr:`layers`."""
        return self._layers

    @property
    def input(self):
        """ Getter for attribute input."""
        return self._input

    @input.setter
    def input(self, value):
        """ Setter for attribute input."""
        self._input = value

    @property
    def forward_requires_grad(self):
        """ Getter for read-only attribute forward_requires_grad"""
        return self._forward_requires_grad

    @property
    def save_df(self):
        """ Getter for read-only attribute :attr:`save_df`."""
        return self._save_df


    def forward(self, x):
        """
        Propagate the input forward through the MLP network.
        Args:
            x: the input to the network
        returns:
            y: the output of the network
        """

        self.input = x
        y = x

        for layer in self.layers:
            y = layer.forward(y)

        if y.requires_grad == False:
            y.requires_grad = True

        return y

    def compute_output_target(self, loss, target_lr):
        """
        Compute the output target.
        Args:
            loss (torch.Tensor): output loss of the network
            target_lr: the learning rate for computing the output target based
                on the gradient of the loss w.r.t. the output layer
        Returns: Mini-batch of output targets
        """

        output_activations = self.layers[-1].activations

        gradient = torch.autograd.grad(loss, output_activations,
                                       retain_graph=self.forward_requires_grad)[0].detach()
        output_targets = output_activations - target_lr*gradient
        return output_targets

    @abstractmethod
    def backward(self, loss, target_lr, save_target):
        """ Send the feedback to all the hidden layers and compute the updates
        for the forward parameters.
        Args:
            loss (torch.Tensor): output loss of the network
            target_lr: the learning rate for computing the output target based
                on the gradient of the loss w.r.t. the output layer
            save_target (bool): flag indicating whether the target activation
                should be
                saved in the layer object for later use.
        """

        pass

    @abstractmethod
    def compute_feedback_gradients(self):
        """
        Compute the updates for the feedback parameters.
        """

        pass


    def get_forward_parameter_list(self):
        """
        Returns: a list with all the forward parameters (weights and biases) of
            the network.
        """

        parameterlist = []
        for layer in self.layers:
            parameterlist.append(layer.weights)
            if layer.use_bias:
                parameterlist.append(layer.bias)
        return parameterlist


    def get_forward_parameter_list_first_layer(self):
        """
        Returns: a list with only the forward parameters of the first layer.
        """

        parameterlist = []
        parameterlist.append(self.layers[0].weights)
        if self.layers[0].use_bias:
            parameterlist.append(self.layers[0].bias)
        return parameterlist

    @abstractmethod
    def get_feedback_parameter_list(self):
        """
        Return a list with all the feedback parameters of the network.
        """

        pass


    def compute_bp_angles(self, loss, i, retain_graph=False):
        """
        Compute the angles of the current forward parameter updates of layer i
        with the backprop update for those parameters.
        Args:
            loss (torch.Tensor): the loss value of the current minibatch.
            i (int): layer index
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.
        Returns (tuple): Tuple containing the angle in degrees between the
            updates for the forward weights at index 0 and the forward bias
            at index 1 (if bias is not None).
        """

        bp_gradients = self.layers[i].compute_bp_update(loss,
                                                        retain_graph)
        gradients = self.layers[i].get_forward_gradients()
        if utils.contains_nan(bp_gradients[0].detach()):
            print('bp update contains nan (layer {}):'.format(i))
            print(bp_gradients[0].detach())
        if utils.contains_nan(gradients[0].detach()):
            print('weight update contains nan (layer {}):'.format(i))
            print(gradients[0].detach())
        if torch.norm(gradients[0].detach(), p='fro') < 1e-14:
            print('norm updates approximately zero (layer {}):'.format(i))
            print(torch.norm(gradients[0].detach(), p='fro'))
            print(gradients[0].detach())
        if torch.norm(gradients[0].detach(), p='fro') == 0:
            print('norm updates exactly zero (layer {}):'.format(i))
            print(torch.norm(gradients[0].detach(), p='fro'))
            print(gradients[0].detach())

        weights_angle = utils.compute_angle(bp_gradients[0].detach(),
                                            gradients[0])
        if self.layers[i].use_bias:
            bias_angle = utils.compute_angle(bp_gradients[1].detach(),
                                             gradients[1])
            return (weights_angle, bias_angle)
        else:
            return (weights_angle, )

    def save_logs(self, writer, step):
        """
        Save logs and plots for tensorboardX.
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
        """

        for i in range(len(self.layers)):
            name = 'layer {}'.format(i+1)
            self.layers[i].save_logs(writer, step, name)


    @abstractmethod
    def save_feedback_batch_logs(self, writer, step, init=False):
        """
        Save the logs for the current minibatch on tensorboardX.
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
            init (bool): flag indicating that the training is in the
                initialization phase (only training the feedback weights).
        """

        pass


    def save_bp_angles(self, writer, step, loss, retain_graph=False,
                       save_tensorboard=True, save_dataframe=True):
        """
        Save the angles of the current forward parameter updates
        with the backprop update for those parameters in the corresponding
        dataframe and in Tensorboard (if ``save_tensorboard=True``).
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
            loss (torch.Tensor): the loss value of the current minibatch.
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.
            save_tensorboard (bool): save the angle results also to Tensorboard.
        """

        layer_indices = range(len(self.layers))

        for i in layer_indices:
            if i != layer_indices[-1]:
                retain_graph_flag = True
            else:
                retain_graph_flag = retain_graph
            angles = self.compute_bp_angles(loss, i, retain_graph_flag)

            if save_tensorboard:
                name = 'layer {}'.format(i+1)
                writer.add_scalar(
                    tag='{}/weight_bp_angle'.format(name),
                    scalar_value=angles[0],
                    global_step=step
                )
                if self.layers[i].use_bias:
                    writer.add_scalar(
                        tag='{}/bias_bp_angle'.format(name),
                        scalar_value=angles[1],
                        global_step=step
                    )

            if save_dataframe:
                self.bp_angles.at[step, i] = angles[0].item()


    def save_nullspace_norm_ratio(self, writer, step, output_activation,
                                  retain_graph=False, save_tensorboard=True,
                                  save_dataframe=True):

        layer_indices = range(len(self.layers))

        for i in layer_indices:
            if i != layer_indices[-1]:
                retain_graph_flag = True
            else:
                retain_graph_flag = retain_graph

            relative_norm = self.layers[i].compute_nullspace_relative_norm(
                output_activation,
                retain_graph=retain_graph_flag
            )
            if save_tensorboard:
                name = 'layer {}'.format(i + 1)

                writer.add_scalar(
                    tag='{}/nullspace_relative_norm'.format(name),
                    scalar_value=relative_norm,
                    global_step=step
                )

            if save_dataframe:
                self.nullspace_relative_norm.at[step, i] = relative_norm.item()


    def save_gnt_angles(self, writer, step, output_activation, loss,
                        damping, steady_state=False, nonlinear=False,
                        retain_graph=False, save_dataframe=True,
                        save_tensorboard=True, custom_result_df=None):
        """
        Compute the angle between the actual weight updates of the model
        (e.g. resulting from TPDI) on the one hand, and the weight updates
        resulting from ideal Gauss-Newton targets for the layers on the
        other hand. Save the angle in the tensorboard X writer
        (if ``save_tensorboard=true``) and in the
        corresponding dataframe (if ``save_dataframe=True``)
        Args:
            writer: Tensorboard writer
            step (int): x-axis index used for tensorboard
            output_activation (torch.Tensor): the post-nonlinearity output activation of the
                network
            loss (torch.Tensor): The training loss
            damping (float): damping used for computing the ideal
                Gauss-Newton targets.
            steady_state (bool): See docstring of
                :func:`networks.TPDI_networks.RTTPNetwork.compute_gnt_parameter_update`
            nonlinear (bool): See docstring of
                :func:`networks.TPDI_networks.RTTPNetwork.compute_gnt_parameter_update`
            retain_graph: Flag indicating whether the Pytorch Autograd
                computational graph should be retained or not.
            save_dataframe (bool): Flag indicating whether a dataframe of the angles
                should be saved in the network object
            save_tensorboard (bool): Flag indicating whether the angles should
                be saved in Tensorboard
            custom_result_df (pd.Dataframe): When different from None, the
                angles will be saved into this dataframe instead of the
                dataframe of the network object.
        """

        output_error = torch.autograd.grad(loss, output_activation,
                                           retain_graph=True)[0].detach()

        gnt_updates = self.compute_gnt_update(output_error=output_error,
                                              damping=damping,
                                              linear=True,
                                              retain_graph=retain_graph)
        gnt_parameter_updates = self.compute_gnt_parameter_update(
            gnt_updates=gnt_updates,
            steady_state=steady_state,
            nonlinear=nonlinear
        )
        for i in range(self.depth):
            parameter_update = self.layers[i].get_forward_gradients()
            if self.use_bias:
                gnt_weight_update_i = gnt_parameter_updates[2*i]
                gnt_bias_update_i = gnt_parameter_updates[2*i+1]
                bias_angle = utils.compute_angle(gnt_bias_update_i,
                                                 parameter_update[1])
            else:
                gnt_weight_update_i = gnt_parameter_updates[i]

            weights_angle = utils.compute_angle(gnt_weight_update_i,
                                                parameter_update[0])

            if save_tensorboard:
                name = 'layer {}'.format(i + 1)
                writer.add_scalar(
                    tag='{}/weight_gnt_angle'.format(name),
                    scalar_value=weights_angle,
                    global_step=step)
                if self.use_bias:
                    writer.add_scalar(
                        tag='{}/bias_gnt_angle'.format(name),
                        scalar_value=bias_angle,
                        global_step=step
                    )
            if custom_result_df is not None:
                custom_result_df.at[step, i] = weights_angle.item()
            elif save_dataframe:
                self.gnt_angles.at[step, i] = weights_angle.item()

        parameter_updates_concat = self.get_vectorized_parameter_updates()
        gnt_updates_concat = utils.vectorize_tensor_list(gnt_parameter_updates)

        total_angle = utils.compute_angle(parameter_updates_concat,
                                          gnt_updates_concat)
        if save_tensorboard:
            writer.add_scalar(
                tag='total_alignment/gnt_angle',
                scalar_value=total_angle,
                global_step=step
            )
        if save_dataframe:
            self.gnt_angles_network.at[step, 0] = total_angle.item()


    def save_gn_angles(self, writer, step, output_activation, loss, damping,
                       retain_graph=False, save_tensorboard=True,
                       save_dataframe=True):
        """
        Compute the angle between the actual weight updates of the model
        (e.g. resulting from TPDI) on the one hand, and the weight updates
        resulting from ideal Gauss-Newton targets for the layers on the
        other hand. Save the angle in the tensorboard X writer
        (if ``save_tensorboard=true``) and in the
        corresponding dataframe (if ``save_dataframe=True``)
        Args:
            writer: Tensorboard writer
            step (int): x-axis index used for tensorboard
            output_activation (torch.Tensor): the post-nonlinearity output activation of the
                network
            loss (torch.Tensor): The training loss
            damping (float): damping used for computing the ideal
                Gauss-Newton targets.
            retain_graph: Flag indicating whether the Pytorch Autograd
                computational graph should be retained or not.
            save_dataframe (bool): Flag indicating whether a dataframe of the angles
                should be saved in the network object
            save_tensorboard (bool): Flag indicating whether the angles should
                be saved in Tensorboard
        """

        output_error = torch.autograd.grad(loss, output_activation,
                                           retain_graph=True)[0].detach()

        gn_parameter_updates = self.compute_gn_parameter_update(
            output_error=output_error,
            damping=damping,
            retain_graph=retain_graph
        )
        for i in range(self.depth):
            parameter_update = self.layers[i].get_forward_gradients()
            if self.use_bias:
                gn_weight_update_i = gn_parameter_updates[2*i]
                gn_bias_update_i = gn_parameter_updates[2*i+1]
                bias_angle = utils.compute_angle(gn_bias_update_i,
                                                 parameter_update[1])
            else:
                gn_weight_update_i = gn_parameter_updates[i]

            weights_angle = utils.compute_angle(gn_weight_update_i,
                                                parameter_update[0])

            if save_tensorboard:
                name = 'layer {}'.format(i + 1)
                writer.add_scalar(
                    tag='{}/weight_gn_angle'.format(name),
                    scalar_value=weights_angle,
                    global_step=step)
                if self.use_bias:
                    writer.add_scalar(
                        tag='{}/bias_gn_angle'.format(name),
                        scalar_value=bias_angle,
                        global_step=step
                    )
            if save_dataframe:
                self.gn_angles.at[step, i] = weights_angle.item()

        parameter_updates_concat = self.get_vectorized_parameter_updates()
        gn_updates_concat = utils.vectorize_tensor_list(gn_parameter_updates)

        total_angle = utils.compute_angle(parameter_updates_concat,
                                          gn_updates_concat)
        if save_tensorboard:
            writer.add_scalar(
                tag='total_alignment/gn_angle',
                scalar_value=total_angle,
                global_step=step
            )
        if save_dataframe:
            self.gn_angles_network.at[step, 0] = total_angle.item()


    def compute_gnt_update(self, output_error, damping=0, linear=True,
                           retain_graph=False):
        """
        Compute the negative of the Gauss-Newton update
        for either the pre-nonlinearity
        activations (if `linear=True`) or post-nonlinearity activations
        (if `linear=False`) using the full Jacobian of the network
        (i.e. computed with `compute_full_jacobian(..)`, using the
        provided output error :math:`\boldsymbol{\delta}_L`.
        .. math::
            \bar{\ve{v}}^{GN} = \bar{J}^T(\bar{J}\bar{J}^T + \lambda I)^{-1} \
            \boldsymbol{\delta}_L
        Args:
            output_error (torch.Tensor): :math:`- \boldsymbol{\delta}_L` in the computation
                of the fixed output target, i.e the difference
                between the feedforward
                output activation and the fixed output target.
            damping (float): damping constant :math:`\lambda`
            linear (bool): Flag indicating whether the full Jacobian :math:`\bar{J}`
                should be computed with respect to the pre-nonlinearity
                activations (if ``True``) or post-nonlinearity activations
                (if ``False``)
            retain_graph (bool): Flag indicating whether the Pytorch Autograd
                computational graph should be retained or not.
        Returns (list): A list of length ``self.depth`` containing at index
            ``i`` a torch.Tensor of dimension :math:`B \times n_l`
            with B the minibatch size and n_l the dimension of layer l,
            containing the Gauss-Newton targets for layer ``i``.
        """

        full_jacobian = self.compute_full_jacobian(linear=linear,
                                                   retain_graph=retain_graph)
        gnt = []
        for b in range(output_error.shape[0]):
            gnt.append(
                utils.compute_damped_gn_update(
                    jacobian=full_jacobian[b,:,:],
                    output_error=output_error[b,:],
                    damping=damping
                ).unsqueeze(0)
            )

        return utils.split_in_layers(self, torch.cat(gnt, dim=0).squeeze())


    def get_vectorized_parameter_updates(self):
        """
        Get a vector with all the vectorized, concatenated parameter updates
        in it.
        """

        parameter_list = self.get_forward_parameter_list()
        return torch.cat([p.grad.view(-1).detach() for p in parameter_list])


    def compute_full_jacobian_old(self, linear, retain_graph):
        """
        DEPRECATED
        Compute the Jacobian of the network output (post-nonlinearity)
         with respect to either the concatenated pre-nonlinearity activations of
         all layers (including the output layer!)
         (i.e. `self.layers[i].linearactivations`) if `linear=True` or the
         concatenated post-nonlinearity activations of all layers if
         `linear=False` (i.e. `self.layers[i].activations`).
        Args:
            linear (bool): Flag indicating whether the Jacobian with respect to the
                pre-nonlinearity activations (`linear=True`) should be taken
                or with respect to the post-nonlinearity activations
                (`linear=False`).
            retain_graph (bool): Flag indicating whether the Pytorch Autograd
                computational graph should be retained or not.

        Returns (torch.Tensor): A :math:`B \times n_L \times \sum_{l=1}^L n_l`
            dimensional tensor,
            with B the minibatch size and n_l the dimension of layer l,
            containing the Jacobian of the network output w.r.t. the
            concatenated activations (pre or post-nonlinearity) of all layers,
            for each minibatch sample
        """

        layer_jacobians = []
        output_activation = self.layers[-1].activations
        for i, layer in enumerate(self.layers):
            if i == self.depth-1:
                retain_graph_flag = retain_graph
            else:
                retain_graph_flag = True
            if linear:
                layer_jacobians.append(utils.compute_batch_jacobian(
                    input=layer.linearactivations,
                    output=output_activation,
                    retain_graph=retain_graph_flag
                ))
            else:
                layer_jacobians.append(utils.compute_batch_jacobian(
                    input=layer.activations,
                    output=output_activation,
                    retain_graph=retain_graph_flag
                ))

        return torch.cat(layer_jacobians, dim=2)


    def compute_full_jacobian(self, linear, retain_graph=False):
        """
        NEW IMPLEMENTATION, WITHOUT USING AUTOGRAD
        MUCH MORE EFFICIENT
        BUT DOES NOT RETAIN GRAPH (BECAUSE IT DOES NOT CREATE IT)
        Compute the Jacobian of the network output (post-nonlinearity)
         with respect to either the concatenated pre-nonlinearity activations of
         all layers (including the output layer!)
         (i.e. `self.layers[i].linearactivations`) if `linear=True` or the
         concatenated post-nonlinearity activations of all layers if
         `linear=False` (i.e. `self.layers[i].activations`).
        Args:
            linear (bool): Flag indicating whether the Jacobian with respect to the
                pre-nonlinearity activations (`linear=True`) should be taken
                or with respect to the post-nonlinearity activations
                (`linear=False`).
            retain_graph (bool): WARNING Pytorch Autograd is not used anymore. This
                flag is keep for backwards compatibility, but any further function which
                expects the graphs to be retained, will not find them and will
                generate an error.
        Returns (torch.Tensor): A :math:`B \times n_L \times \sum_{l=1}^L n_l`
            dimensional tensor,
            with B the minibatch size and n_l the dimension of layer l,
            containing the Jacobian of the network output w.r.t. the
            concatenated activations (pre or post-nonlinearity) of all layers,
            for each minibatch sample
        """

        L = self.depth
        layer_jacobians = [None for i in range(self.depth)]
        vectorized_nonlinearity_derivative = [l.compute_vectorized_jacobian(l.linearactivations) for l in self.layers]
        output_activation = self.layers[-1].activations
        batch_size = output_activation.shape[0]
        output_size = output_activation.shape[1]

        layer_jacobians[-1] = \
            torch.eye(output_size).repeat(batch_size, 1, 1).reshape(batch_size, output_size, output_size)
        if linear:
            layer_jacobians[-1] = vectorized_nonlinearity_derivative[-1].view(batch_size, output_size, 1) \
                                  * layer_jacobians[-1]

        for i in range(L-1-1, 0 - 1, -1):
            if linear:
                layer_jacobians[i] = vectorized_nonlinearity_derivative[i].unsqueeze(1) * \
                                     layer_jacobians[i+1].matmul(self.layers[i+1].weights)
            else:
                layer_jacobians[i] = (layer_jacobians[i + 1] * vectorized_nonlinearity_derivative[i + 1].unsqueeze(1))\
                                     .matmul(self.layers[i + 1].weights)

        return torch.cat(layer_jacobians, dim=2)


    def compute_gnt_parameter_update(self, gnt_updates, steady_state=False,
                                     nonlinear=False):
        r"""
        Compute the weight (and bias) updates for each layer,
        resulting from negative of the ideal
        (damped) Gauss-Newton targets for the network layers (computed
        with ``compute_gnt_update()``). When ``steady_state=True`` and
        ``nonlinear=False``, the
        layer activations :math:`\mathbf{r}_{i-1,\text{ss}}`
        at steady state are used for
        computing the weight update:

        ..math::
            \Delta W_i^{GNT,\text{ss}} = \
            \Delta\ve{v}_i^{GN}\ve{r}_{i-1,\text{ss}}^T

        When ``steady_state=False`` and
        ``nonlinear=False``, the
        layer activations :math:`\mathbf{r}_{i-1}^-` at the feedforward phase
        are used for computing the weight update:

        ..math::
            \Delta W_i^{GNT} =  \Delta\ve{v}_i^{GN}(\ve{r}_{i-1}^-)^T

        Note that :math:`\Delta\ve{v}_i^{GN}` is always computed using the
        feedforward activations for evaluating the Jacobian.
        These two weight updates can be adapted to the nonlinear weight update
        variant by putting ``nonlinear=True``, resulting in

        ..math::
            \Delta W_i^{MGNT,\text{ss}} = \big(\phi(\ve{v}_i^- + \
             \Delta \ve{v}_i^{GN}) - \phi(\ve{v}_i^-)\big) \ve{r}_{i-1,\text{ss}}^T

        when ``steady_state=True``, with :math:`\phi` the nonlinear activation
         function and
        ..math::
            \Delta W_i^{MGNT} =  \big(\phi(\ve{v}_i^- + \Delta \ve{v}_i^{GN}) \
             - \phi(\ve{v}_i^-)\big) (\ve{r}_{i-1}^-)^T

        when ``steady_state=False``

        The bias updates are computed by exactly the same formulas, only
        without the :math:`\ve{r}_{i-1}^T` term at the end.

        Args:
            steady_state (bool): Flag indicating whether the steady-state
                values should be taken to compute the weight update.
            nonlinear (bool): Flag indicating whether the nonlinear variant
                of the weight update rule should be used, as explained above.
            gnt_updates (list): The output of :func:`compute_gnt_update`
                Important: The gnt_updates should be computed for the
                pre-nonlinearity activations, i.e. the ``linear`` argument of
                :func:`compute_gnt_update` must be ``True``!!

        Returns (list):
            - If the network uses biases:
                A list of length ``2*self.depth`` containing the weight
                update for layer ``i`` at index ``2*i`` and the bias update
                at index ``2*i+1``
            - If the network does not use biases:
                A list of length ``self.depth`` containing the weight
                update for layer ``i`` at index ``i``.
        """

        batch_size = self.layers[0].activations.shape[0]
        parameter_updates = []

        for i, layer in enumerate(self.layers):
            if i == 0:
                r_previous = self.input
            else:
                if steady_state:
                    r_previous = self.layers[i-1].activations_ss
                else:
                    r_previous = self.layers[i-1].activations

            if nonlinear:
                activation_updates = layer.forward_activationfunction(
                    layer.linearactivations + gnt_updates[i]) - \
                    layer.forward_activationfunction(layer.linearactivations)
            else:
                activation_updates = gnt_updates[i]
            weight_update = 1./batch_size * activation_updates.t().mm(r_previous)
            parameter_updates.append(weight_update)
            if self.use_bias:
                bias_update = activation_updates.mean(0)
                parameter_updates.append(bias_update)

        return parameter_updates

    def compute_full_weight_jacobian(self, retain_graph):
        r"""
        Compute the Jacobian of the network output (post-nonlinearity)
         with respect to the concatenated vectorized weights and biases.

         ..math::
             J_{\bar{W}} &= \frac{\partial \mathbf{r}_L}{\partial \bar{W}} \\
             \bar{W} &= [vec(W_1)^T \mathbf{b}_1^T ... vec(W_L)^T \mathbf{b}_L^T]^T

         If the minibatch size is bigger than 1, the jacobian for each batch
         sample is computed and concatenated along dimension 0.
        Args:
            retain_graph (bool): Flag indicating whether the Pytorch Autograd
                computational graph should be retained or not.

        Returns (torch.Tensor): A
        :math:`B \times n_L \times \sum_{l=1}^L (n_l + 1)n_{l-1}`
            dimensional tensor if the network contains biases, otherwise
            a :math:`B \times n_L \times \sum_{l=1}^L (n_l)n_{l-1}`
            dimensional tensor,
            with B the minibatch size and n_l the dimension of layer l,
            containing the Jacobian of the network output w.r.t. the
            concatenated vectorized weights (and biases),
            for each minibatch sample
        """

        net_parameters = self.get_forward_parameter_list()

        batch_jacobians = []

        output_activations = self.layers[-1].activations
        for b in range(output_activations.shape[0]):
            if b == output_activations.shape[0] - 1:
                retain_graph_flag = retain_graph
            else:
                retain_graph_flag = True

            output_activation = output_activations[b, :]
            batch_jacobians.append(
                utils.compute_jacobian(net_parameters, output_activation,
                                       retain_graph=retain_graph_flag).unsqueeze(0)
            )

        return torch.cat(batch_jacobians, dim=0)


    def compute_gn_parameter_update(self, output_error, damping=0,
                                    retain_graph=False):
        r"""
        Compute the negative of the Gauss-Newton weight updates according to

        ..math::
            \Delta \bar{W}^{GN} = \frac{1}{B} \sum_{b=1}^B J_{|bar{W}}^{(b)T} \
            (J_{|bar{W}}^{(b)} J_{|bar{W}}^{(b)T} + \lambda I)^{-1} \
            \boldsymbol{\delta}_L^{(b)}

        with b indicating the batch sample and :math:`\bar{W}` as defined in
        :func:`compute_full_weight_jacobian`, hence containing the concatenated
        vectorized weights and biases.

        Args:
            output_error (torch.Tensor): :math:`-\boldsymbol{\delta}_L` in the computation
                of the fixed output target, i.e the difference
                between the feedforward
                output activation and the fixed output target.
            damping (float): damping constant :math:`\lambda`
            retain_graph (bool): Flag indicating whether the Pytorch Autograd
                computational graph should be retained or not.

        Returns (list): a list of size 2*``self.depth`` if the network has
            biases, otherwise a list of size ``self.depth``, which contains
            the Gauss-Newton updates for the weights (and biases) of the
            network. The updates are in the original shape of the parameters
            (e.g. not vectorized).
        """

        full_weight_jacobians = self.compute_full_weight_jacobian(
            retain_graph=retain_graph)
        parameter_update = torch.zeros(full_weight_jacobians.shape[2],1)
        batchsize = full_weight_jacobians.shape[0]
        for b in range(batchsize):
            parameter_update += 1. / batchsize * utils.compute_damped_gn_update(
                jacobian=full_weight_jacobians[b, :, :],
                output_error=output_error[b, :],
                damping=damping
            )

        return utils.split_and_reshape_vectorized_parameters(self, parameter_update)


    def save_jacobian_transpose_angle(self, writer, step, retain_graph=False,
                                      save_tensorboard=True,
                                      save_dataframe=True, init=False):
        """
        Compute and save the angle of the direct feedback weights with the
        transpose of the full Jacobian of the network output with respect to the
        layer activations.
        Args:
            writer: Tensorboard writer
            step (int): x-axis index used for tensorboard
            retain_graph: Flag indicating whether the Pytorch Autograd
                computational graph should be retained or not.
            save_dataframe (bool): Flag indicating whether a dataframe of the angles
                should be saved in the network object
            save_tensorboard (bool): Flag indicating whether the angles should
                be saved in Tensorboard
            init (bool): Flag indicating whether we are in pre-training
        """

        assert hasattr(self, 'full_Q')
        full_jac_minibatch = self.compute_full_jacobian(linear=True,
                                              retain_graph=retain_graph)
        batchsize = full_jac_minibatch.shape[0]
        average_angle = 0.
        for i in range(batchsize):
            full_jac = full_jac_minibatch[i,:,:]
            angle = utils.compute_angle(full_jac.T, self.full_Q)
            average_angle += angle/batchsize

        if save_tensorboard:
            if init:
                name='feedback_training/jacobian_transpose_angle_init'
            else:
                name = 'feedback_training/jacobian_transpose_angle'
            writer.add_scalar(
                tag=name,
                scalar_value=average_angle,
                global_step=step
            )

        if save_dataframe:
            if init:
                self.jac_transpose_angles_init.at[step, 0] = average_angle.item()
            else:
                self.jac_transpose_angles.at[step, 0] = average_angle.item()


    def save_jacobian_pinv_angle(self, writer, step, retain_graph=False,
                                      save_tensorboard=True,
                                      save_dataframe=True,
                                      damping=0.,
                                      init=False):
        """
        Compute and save the angle of the direct feedback weights with the
        pseudoinverse of the full Jacobian of the network output with respect to the
        layer activations.
        Args:
            writer: Tensorboard writer
            step (int): x-axis index used for tensorboard
            retain_graph: Flag indicating whether the Pytorch Autograd
                computational graph should be retained or not.
            save_dataframe (bool): Flag indicating whether a dataframe of the angles
                should be saved in the network object
            save_tensorboard (bool): Flag indicating whether the angles should
                be saved in Tensorboard
            damping (float): damping constant :math:`\lambda` for computing
                the damped pseudoinverse according to
                :math:`J^{\dagger} = J^T(JJ^T + \lambda I)^{-1}`
            init (bool): flag indicating whether we are in pre-training or not
        """

        assert hasattr(self,
                       'full_Q')
        full_jac_minibatch = self.compute_full_jacobian(linear=True,
                                                        retain_graph=retain_graph)
        batchsize = full_jac_minibatch.shape[0]
        average_angle = 0.
        for i in range(batchsize):
            jacobian = full_jac_minibatch[i, :, :]
            if damping == 0:
                full_jac_pinv = torch.pinverse(jacobian)
            else:
                if jacobian.shape[0] >= jacobian.shape[1]:
                    G = jacobian.t().mm(jacobian)
                    C = G + damping * torch.eye(G.shape[0])
                    full_jac_pinv = torch.inverse(C).mm(jacobian.t())
                else:
                    G = jacobian.mm(jacobian.t())
                    C = G + damping * torch.eye(G.shape[0])
                    full_jac_pinv = jacobian.t().mm(torch.inverse(C))
            angle = utils.compute_angle(full_jac_pinv, self.full_Q)
            average_angle += angle / batchsize

        if save_tensorboard:
            if init:
                name = 'feedback_training/jacobian_pinv_angle_init'
            else:
                name = 'feedback_training/jacobian_pinv_angle'
            writer.add_scalar(
                tag=name,
                scalar_value=average_angle,
                global_step=step
            )

        if save_dataframe:
            if init:
                self.jac_pinv_angles_init.at[step, 0] = average_angle.item()
            else:
                self.jac_pinv_angles.at[step, 0] = average_angle.item()


    def save_gn_gnt_angles(self, writer, step, output_activation, loss,
                           damping, retain_graph=False, save_tensorboard=True,
                           save_dataframe=True, steady_state=False,
                           nonlinear=False):
        """
        Compute and save the angle between the GN parameter update and the GNT
        parameter update.
        Args:
            (....): see args of
                :func:`networks.abstract_network.AbstractNetwork.save_gnt_angles`
                and
                :func:`networks.abstract_network.AbstractNetwork.save_gn_angles`
        """

        output_error = torch.autograd.grad(loss, output_activation,
                                           retain_graph=True)[0].detach()

        gnt_updates = self.compute_gnt_update(output_error=output_error,
                                              damping=damping,
                                              linear=True,
                                              retain_graph=True)
        gnt_parameter_updates = self.compute_gnt_parameter_update(
            gnt_updates=gnt_updates,
            steady_state=steady_state,
            nonlinear=nonlinear
        )

        gn_parameter_updates = self.compute_gn_parameter_update(
            output_error=output_error,
            damping=damping,
            retain_graph=retain_graph
        )

        for i in range(self.depth):
            if self.use_bias:
                gnt_weight_update_i = gnt_parameter_updates[2 * i]
                gnt_bias_update_i = gnt_parameter_updates[2 * i + 1]
                gn_weight_update_i = gn_parameter_updates[2 * i]
                gn_bias_update_i = gn_parameter_updates[2 * i + 1]
                bias_angle = utils.compute_angle(gnt_bias_update_i,
                                                 gn_bias_update_i)
            else:
                gnt_weight_update_i = gnt_parameter_updates[i]
                gn_weight_update_i = gn_parameter_updates[i]

            weights_angle = utils.compute_angle(gnt_weight_update_i,
                                                gn_weight_update_i)

            if save_tensorboard:
                name = 'layer {}'.format(i + 1)
                writer.add_scalar(
                    tag='{}/weight_gnt_gn_angle'.format(name),
                    scalar_value=weights_angle,
                    global_step=step)
                if self.use_bias:
                    writer.add_scalar(
                        tag='{}/bias_gnt_gn_angle'.format(name),
                        scalar_value=bias_angle,
                        global_step=step
                    )


    def save_bp_gnt_angles(self, writer, step, output_activation, loss,
                           damping, retain_graph=False, save_tensorboard=True,
                           save_dataframe=True, steady_state=False,
                           nonlinear=False):
        """
        Compute and save the angle between the BP parameter update and the GNT
        parameter update.
        Args:
            (....): see args of
                :func:`networks.abstract_network.AbstractNetwork.save_gnt_angles`
                and
                :func:`networks.abstract_network.AbstractNetwork.save_bp_angles`
        """

        output_error = torch.autograd.grad(loss, output_activation,
                                           retain_graph=True)[0].detach()

        gnt_updates = self.compute_gnt_update(output_error=output_error,
                                              damping=damping,
                                              linear=True,
                                              retain_graph=True)
        gnt_parameter_updates = self.compute_gnt_parameter_update(
            gnt_updates=gnt_updates,
            steady_state=steady_state,
            nonlinear=nonlinear
        )

        for i, layer in enumerate(self.layers):
            if i < self.depth-1:
                retain_graph_flag = True
            else:
                retain_graph_flag = retain_graph
            bp_gradients = layer.compute_bp_update(loss, retain_graph_flag)
            if self.use_bias:
                gnt_weight_update_i = gnt_parameter_updates[2 * i]
                gnt_bias_update_i = gnt_parameter_updates[2 * i + 1]
                bias_angle = utils.compute_angle(gnt_bias_update_i,
                                                 bp_gradients[1].detach())
            else:
                gnt_weight_update_i = gnt_parameter_updates[i]

            weights_angle = utils.compute_angle(gnt_weight_update_i,
                                                bp_gradients[0].detach())

            if save_tensorboard:
                name = 'layer {}'.format(i + 1)
                writer.add_scalar(
                    tag='{}/weight_gnt_bp_angle'.format(name),
                    scalar_value=weights_angle,
                    global_step=step)
                if self.use_bias:
                    writer.add_scalar(
                        tag='{}/bias_gnt_bp_angle'.format(name),
                        scalar_value=bias_angle,
                        global_step=step
                    )


    def save_gnt_ss_no_ss_angles(self, writer, step, output_activation, loss,
                           damping, retain_graph=False, save_tensorboard=True,
                           save_dataframe=True, steady_state=False,
                           nonlinear=False):
        """
        Compute and save the angle between the GNT parameter update where
        the steady state value of :math:`r_{i-1}` is used and the GNT parameter
        update where the initial feedforward activation of :math:`r_{i-1}` is
        used.
        Args:
            (....): see args of
                :func:`networks.abstract_network.AbstractNetwork.save_gnt_angles`
                and
                :func:`networks.abstract_network.AbstractNetwork.save_gn_angles`
        """

        output_error = torch.autograd.grad(loss, output_activation,
                                           retain_graph=True)[0].detach()

        gnt_updates = self.compute_gnt_update(output_error=output_error,
                                              damping=damping,
                                              linear=True,
                                              retain_graph=retain_graph)
        gnt_parameter_updates_ss = self.compute_gnt_parameter_update(
            gnt_updates=gnt_updates,
            steady_state=True,
            nonlinear=nonlinear
        )

        gnt_parameter_updates_no_ss = self.compute_gnt_parameter_update(
            gnt_updates=gnt_updates,
            steady_state=False,
            nonlinear=nonlinear
        )


        for i in range(self.depth):
            if self.use_bias:
                gnt_ss_weight_update_i = gnt_parameter_updates_ss[2 * i]
                gnt_ss_bias_update_i = gnt_parameter_updates_ss[2 * i + 1]
                gnt_weight_update_i = gnt_parameter_updates_no_ss[2 * i]
                gnt_bias_update_i = gnt_parameter_updates_no_ss[2 * i + 1]
                bias_angle = utils.compute_angle(gnt_bias_update_i,
                                                 gnt_ss_bias_update_i)
            else:
                gnt_ss_weight_update_i = gnt_parameter_updates_ss[i]
                gnt_weight_update_i = gnt_parameter_updates_no_ss[i]

            weights_angle = utils.compute_angle(gnt_weight_update_i,
                                                gnt_ss_weight_update_i)

            if save_tensorboard:
                name = 'layer {}'.format(i + 1)
                writer.add_scalar(
                    tag='{}/weight_gnt_ss_no_ss_angle'.format(name),
                    scalar_value=weights_angle,
                    global_step=step)
                if self.use_bias:
                    writer.add_scalar(
                        tag='{}/bias_gnt_ss_no_ss_angle'.format(name),
                        scalar_value=bias_angle,
                        global_step=step
                    )


    def contains_nans(self):
        """
        Check whether the network parameters contain NaNs.
        Returns (bool): Flag indicating whether the network contains a NaN.
        Also returns True if some parameters are above 1000, which indicates
        divergence
        """

        threshold = 1000
        for p in self.get_forward_parameter_list():
            if utils.contains_nan(p) or torch.sum(p > threshold) > 0:
                return True

        for p in self.get_feedback_parameter_list():
            if utils.contains_nan(p) or torch.sum(p > threshold) > 0:
                return True

        return False


    def zero_grad(self):
        """
        Initialize all the gradients of the network parameters
        (both forward and feedback) to zero.
        """
        
        for param in self.get_forward_parameter_list():
            param.grad = torch.zeros_like(param)

        for param in self.get_feedback_parameter_list():
            param.grad = torch.zeros_like(param)