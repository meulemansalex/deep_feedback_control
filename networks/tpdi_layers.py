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

from networks.abstract_layer import AbstractLayer
import torch


class DFCLayer(AbstractLayer):

    def __init__(self, in_features, out_features, size_output, bias=True,
                 forward_requires_grad=False, forward_activation='tanh',
                 initialization='orthogonal', clip_grad_norm=-1):

        super().__init__(in_features=in_features,
                         out_features=out_features,
                         bias=bias,
                         forward_requires_grad=forward_requires_grad,
                         forward_activation=forward_activation,
                         initialization=initialization)

        self._clip_grad_norm = clip_grad_norm

        self._feedback_weights = torch.zeros(out_features, size_output)

    @property
    def feedbackweights(self):
        """Getter for read-only attribute :attr:`inversion_matrix` (ie K)"""
        return self._feedback_weights

    @property
    def clip_grad_norm(self):
        """Getter for read-only attribute :attr:`clip_grad_norm`"""
        return self._clip_grad_norm

    @feedbackweights.setter
    def feedbackweights(self, tensor):
        """ Setter for feedbackweights"""
        self._feedback_weights = tensor


    def compute_forward_gradients(self, h_target, h_previous):
        raise NotImplementedError(
            "TODO: move :func:`compute_forward_gradinets_deltav`"
                                  "to this method")


    def compute_forward_gradients_deltav(self, delta_v, r_previous,
                                         learning_rule="voltage_difference",
                                         scale=1., saving_ndi_updates=False):
        """
        Computes gradients using a local-in-time learning rule.
        There are three possible learning rules to be applied:
        * voltage_difference : difference between basal and somatic voltage
        * derivative_matrix : the derivative matrix (of somatic voltage wrt
        forward weights) is added to the previous rule
        * nonlinear_difference: by Taylor expansion, the difference between the
        nonlinear transformation of basal and
        somatic voltages is approximately equal to the addition of the
        derivative matrix. Indeed, second and third
        learning rules have been found to be completely equivalent in practice.

        New parameter added: "saving_ndi_updates". When False, computed updates
        are added to weights.grad and bias.grad, to be later updated. When True,
        computed updates are added to ndi_updates_weights/bias, to later compare
        with the ss/continuous updates.
        """

        if learning_rule == "voltage_difference":
            teaching_signal = 2 * (-delta_v)

        elif learning_rule == "derivative_matrix":
            vectorized_jacobians = self.compute_vectorized_jacobian(self.linearactivations)
            teaching_signal = -2 * vectorized_jacobians * delta_v

        elif learning_rule == "nonlinear_difference":
            v_basal = torch.matmul(r_previous, self.weights.t())
            if self.bias is not None:
                v_basal += self.bias.unsqueeze(0).expand_as(v_basal)
            v_somatic = delta_v + v_basal 
            teaching_signal = -2*(self.forward_activationfunction(v_somatic) - self.forward_activationfunction(v_basal))

        else:
            raise ValueError("The specified learning rule %s is not valid" % learning_rule)

        batch_size = r_previous.shape[0]
        bias_grad = teaching_signal.mean(0)

        weights_grad = 1./batch_size * teaching_signal.t().mm(r_previous)

        if saving_ndi_updates:
            if self.bias is not None:
                self.ndi_updates_bias = scale * bias_grad.detach()
                if self.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.ndi_updates_bias, max_norm=self.clip_grad_norm)
            self.ndi_updates_weights = scale * weights_grad.detach()
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.ndi_updates_weights, max_norm=self.clip_grad_norm)

        else:
            if self.bias is not None:
                self._bias.grad += scale * bias_grad.detach()
                if self.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self._bias, max_norm=self.clip_grad_norm)
            self._weights.grad += scale * weights_grad.detach()
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self._weights, max_norm=self.clip_grad_norm)


    def compute_forward_gradients_deltav_continuous(self, vs_time, vb_time, r_previous_time, t_start=None, t_end=None, learning_rule="voltage_difference"):
        """
        Computes gradients using an integration (sum) of voltage differences across comparments.
        It is a local-in-time learning rule. There are three possible learning rules to be applied:
        * voltage_difference : difference between basal and somatic voltage
        * derivative_matrix : the derivative matrix (of somatic voltage wrt forward weights) is added to the previous rule
        * nonlinear_difference: by Taylor expansion, the difference between the nonlinear transformation of basal and
        somatic voltages is approximately equal to the addition of the derivative matrix. Indeed, second and third
        learning rules have been found to be completely equivalent in practice.
        """

        if t_start is None: t_start = 0
        if t_end is None: t_end = vs_time.shape[0]
        T = t_end - t_start
        batch_size = r_previous_time.shape[1]

        if learning_rule == "voltage_difference":
            time_diff = vs_time[t_start:t_end] - vb_time[t_start:t_end]
            bias_grad = -2 * 1./T * torch.sum(time_diff, axis=0).mean(0)
            weights_grad = -2 * 1./batch_size * 1./T * torch.sum(time_diff.permute(0,2,1) @ r_previous_time[t_start:t_end], axis=0)

        elif learning_rule == "derivative_matrix":
            vectorized_jacobians = self.compute_vectorized_jacobian(self.linearactivations)
            time_diff = vs_time[t_start:t_end] - vb_time[t_start:t_end]
            bias_grad = -2 * 1. / T * torch.sum(vectorized_jacobians * time_diff, axis=0).mean(0)
            weights_grad = -2 * 1. / batch_size * 1. / T * \
                           torch.sum((vectorized_jacobians * time_diff).permute(0, 2, 1) @ r_previous_time[t_start:t_end], axis=0)

        elif learning_rule == "nonlinear_difference":
            time_diff = self.forward_activationfunction(vs_time[t_start:t_end]) \
                        - self.forward_activationfunction(vb_time[t_start:t_end])
            bias_grad = -2 * 1. / T * torch.sum(time_diff, axis=0).mean(0)
            weights_grad = -2 * 1. / batch_size * 1. / T * torch.sum(time_diff.permute(0, 2, 1) \
                                                                     @ r_previous_time[t_start:t_end], axis=0)

        else:
            raise ValueError("The specified learning rule %s is not valid" % learning_rule)

        if self.bias is not None:
            self._bias.grad = bias_grad.detach()
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self._bias, max_norm=self.clip_grad_norm)
        self._weights.grad = weights_grad.detach()
        if self.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self._weights, max_norm=self.clip_grad_norm)


    def compute_feedback_gradients(self, v_a, u,
                                   beta, sigma, scale=1.):
        r"""
        Compute the gradients for the feedback parameters Q (in the code they
        are presented by ``self.feedbackweights``) and save it in
        ``self._feedbackweights.grad``, using the
        following update rule:

        ..math::
            \Delta Q_i = \frac{1}{\sigma^2} \mathbf{v}^A_i \mathbf{u}^T \
            - \beta Q_i

        :math:`\beta` indicates the homeostatic scaling, which is equal to
        :math:`\beta = b(||Q||_F^2 - c)` with b the weight decay and c the
        homeostatic constant. If no homeostatic weight decay is wanted, but
        normal weight decay instead, put beta on zero and use the cmd arg
        feedback_wd to introduce normal weight decay (and cmd arg
        use_homeostatic_wd_fb should be on False of course).

        Note that pytorch saves the positive gradient, hence we should save
        :math:`-\Delta Q_i`.

        Args:
            delta_v (torch.Tensor): :math:`\Delta \mathbf{v}_i`, the apical compartment voltage
                at its initial perturbed state.
            u: :math:`\mathbf{u}`, the control input at its initial state
            beta (float): homeostatic weight decay parameter.
            sigma (float): the standard deviation of the noise used for
                obtaining an output target in the feedback training
            scale (float): scaling factor for the gradient. Is used when
                continuous feedback updates are computed in the efficient
                controller implementation, where scale=1/t_max, as we want
                to average over all timesteps.
        """

        batch_size = v_a.shape[0]

        if sigma < 0.01:
            scale = scale/0.01**2
        else:
            scale = scale/sigma**2

        weights_grad = - scale/batch_size * v_a.t().mm(u)
        weights_grad += beta * self.feedbackweights

        self._feedback_weights.grad += weights_grad.detach()


    def compute_feedback_gradients_old(self, delta_v, u,
                                   homeostatic_scaling, sigma):
        r"""
        Compute the gradients for the feedback parameters Q (in the code they
        are presented by ``self.K``) and save it in
        ``self._inversion_matrix.grad``, using the
        following update rule:

        ..math::
            \Delta Q_i = -\frac{1}{\sigma^2} \Delta \mathbf{v}_i \mathbf{u}^T \
            - \beta (\|\bar{Q}\|_F^2 - C) Q

        Note that pytorch saves the positive gradient, hence we should save
        :math:`-\Delta Q_i`.

        Args:
            delta_v (torch.Tensor): :math:`\Delta \mathbf{v}_i`, the apical compartment voltage
                level at steady-state.
            u: :math:`\mathbf{u}`, the control input at steady state
            homeostatic_scaling (float): :math:`\beta (\|\bar{Q}\|_F^2 - C)`, a
                homeostatic scaling constant for the weight decay component of
                the update rule, which pulls the magnitude of all feedback
                weights towards a certain constant :math:`C`.
            sigma (float): the standard deviation of the noise used for
                obtaining an output target in the feedback training
        """

        batch_size = delta_v.shape[0]

        if sigma < 0.01:
            scale = 1/0.01**2
        else:
            scale = 1/sigma**2

        weights_grad = scale/batch_size * delta_v.t().mm(u)
        weights_grad += homeostatic_scaling * self.feedbackweights

        if self._feedback_weights.grad is None:
            self._feedback_weights.grad = weights_grad.detach()
        else:
            self._feedback_weights.grad += weights_grad.detach()


    def compute_feedback_gradients_continuous(self, va_time, u_time,
                                              t_start=None, t_end=None,
                                              sigma=1., beta=0.):
        r"""
        Computes the feedback gradients using an integration (sum) of the
        continuous time learning rule over the specified time interval.
        ..math::
            \frac{dQ_i}{dt} = -v_i^A u^T - \beta Q_i

        :math:`\beta` indicates the homeostatic scaling, which is equal to
        :math:`\beta = b(||Q||_F^2 - c)` with b the weight decay and c the
        homeostatic constant. If no homeostatic weight decay is wanted, but
        normal weight decay instead, put beta on zero and use the cmd arg
        feedback_wd to introduce normal weight decay (and cmd arg
        use_homeostatic_wd_fb should be on False of course).

        The gradients are stored in ``self.feedbackweights.grad``. Important,
        the ``.grad`` attribute represents the positive gradient, hence, we
        need to save :math:`-\Delta Q_i` in it.
        Args:
            va_time (torch.Tensor): The apical compartment voltages over
                a certain time period
            u_time (torch.Tensor): The control inputs over  certain time period
            t_start (torch.Tensor): The start index from which the summation
                over time should start
            t_end (torch.Tensor): The stop index at which the summation over
                time should stop
            sigma (float): the standard deviation of the noise in the network
                dynamics. This is used to scale the fb weight update, such that
                its magnitude is independent of the noise variance.
            beta (float): The homeostatic weight decay parameter
        """

        if t_start is None: t_start = 0 
        if t_end is None: t_end = va_time.shape[0]
        T = t_end - t_start
        batch_size = va_time.shape[1]

        if sigma < 0.01:
            scale = 1 / 0.01 ** 2
        else:
            scale = 1 / sigma ** 2

        feedbackweights_grad = scale/(T * batch_size) * \
                torch.sum(va_time[t_start:t_end].permute(0,2,1) @ u_time[t_start:t_end], axis=0)
        feedbackweights_grad += beta * self.feedbackweights

        self._feedback_weights.grad = feedbackweights_grad.detach()


    def noisy_dummy_forward(self, x, noise):
        """
        Propagates the layer input ``x`` to the next layer, while adding
        ``noise`` to the linear activations of this layer. This method does
        not save the resulting activations and linear activations in the layer
        object.
        Args:
            x (torch.Tensor): layer input
            noise (torch.Tensor): Noise to be added to the linear activations
                of this layer.

        Returns (torch.Tensor): The resulting post-nonlinearity activation of
            this layer.
        """

        a = x.mm(self.weights.t())
        if self.bias is not None:
            a += self.bias.unsqueeze(0).expand_as(a)
        a += noise
        h = self.forward_activationfunction(a)
        return h

    def save_feedback_batch_logs(self, writer, step, name, no_gradient=False,
                                 init=False):
        """
        Save logs for one minibatch.
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
            name (str): The name of the layer
            no_gradient (bool): flag indicating whether we should skip saving
                the gradients of the feedback weights.
            init (bool): flag indicating that the training is in the
                initialization phase (only training the feedback weights).
        """

        if init:
            prefix = 'feedback_training/{}/'.format(name)
        else:
            prefix = name + '/'
        feedback_weights_norm = torch.norm(self.feedbackweights)
        writer.add_scalar(tag=prefix + 'feedback_weights_norm',
                          scalar_value=feedback_weights_norm,
                          global_step=step)
        if self.feedbackweights.grad is not None:
            feedback_weights_gradients_norm = torch.norm(self.feedbackweights.grad)
            writer.add_scalar(
                tag=prefix + 'feedback_weights_gradient_norm',
                scalar_value=feedback_weights_gradients_norm,
                global_step=step)



class TPDILayer(DFCLayer):

    def compute_forward_gradients(self, h_target, h_previous, norm_ratio=1.):
        """
        Compute the gradient of the forward weights and bias, based on the
        local mse loss between the layer activation and layer target.
        The gradients are saved in the .grad attribute of the forward weights
        and forward bias.
        Args:
            h_target (torch.Tensor): the DTP target of the current layer
            h_previous (torch.Tensor): the rate activation of the previous
                layer
            norm_ratio (float): Depreciated.
        """
        
        if self.forward_activation == 'linear':
            teaching_signal = 2 * (self.activations - h_target)
        else:
            vectorized_jacobians = self.compute_vectorized_jacobian(self.linearactivations)
            teaching_signal = 2 * vectorized_jacobians * (self.activations - h_target)
        batch_size = h_target.shape[0]
        bias_grad = teaching_signal.mean(0)
        weights_grad = 1./batch_size * teaching_signal.t().mm(h_previous)

        if self.bias is not None:
            self._bias.grad = bias_grad.detach()
            torch.nn.utils.clip_grad_norm_(self._bias, max_norm=1)
        self._weights.grad = weights_grad.detach()
        torch.nn.utils.clip_grad_norm_(self._weights, max_norm=1)