# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""One-line documentation for sdprop module.

sdprop algorithm

A detailed description of sdprop.

moving_average = decay * moving_average + (1-decay) * gradient
moving_variance = decay * moving_variance + 
                   decay * (1-decay) * (gradient - moving_average) ** 2
Delta = learning_rate * gradient / sqrt(moving_variance-epsilon)

mu <- gamma * mu_{t-1} + (1-gamma) * grad
mom <- gamma * mom_{t-1} + gamma * (1-gamma) * (grad - mu)**2
var <- var - lr * grad / sqrt(mom + epsilon)

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops


class SDPropOptimizer(optimizer.Optimizer):
  """Optimizer that implements the SDProp algorithm.

  See the [paper](http://...).
  """

  def __init__(self,
               learning_rate=0.001,
               gamma=0.99,
               epsilon=1e-7,
               use_locking=False,
               name="SDProp"):
    """Construct a new SDProp optimizer.

    Note that in the dense implementation of this algorithm, variables and their
    corresponding accumulators (momentum, gradient moving average, square
    gradient moving average) will be updated even if the gradient is zero
    (i.e. accumulators will decay, momentum will be applied). The sparse
    implementation (used when the gradient is an `IndexedSlices` object,
    typically because of `tf.gather` or an embedding lookup in the forward pass)
    will not update variable slices or their accumulators unless those slices
    were used in the forward pass (nor is there an "eventual" correction to
    account for these omitted updates). This leads to more efficient updates for
    large embedding lookup tables (where most of the slices are not accessed in
    a particular graph execution), but differs from the published algorithm.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning rate.
      gamma: Discounting factor for the history/coming gradient
      epsilon: Small value to avoid zero denominator.
      use_locking: If True use locks for update operation.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "SDProp".
    """
    super(SDPropOptimizer, self).__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._gamma = gamma
    self._epsilon = epsilon
    # self._centered = centered

    # Tensors for learning rate and momentum.  Created in _prepare.
    self._learning_rate_tensor = None
    self._gamma_tensor = None
    self._epsilon_tensor = None

  def _create_slots(self, var_list):
    for v in var_list:
      init_sd = init_ops.ones_initializer(dtype=v.dtype)
      init_mom = init_ops.ones_initializer(dtype=v.dtype)
      self._get_or_make_slot_with_initializer(v, init_sd, v.get_shape(),
                                              v.dtype, "sd", self._name)
      self._get_or_make_slot_with_initializer(v, init_mom, v.get_shape(),
                                              v.dtype, "mom", self._name)
      # if self._centered:
      #   self._zeros_slot(v, "mg", self._name)
      # self._zeros_slot(v, "momentum", self._name)

  def _prepare(self):
    self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate,
                                                       name="learning_rate")
    self._gamma_tensor = ops.convert_to_tensor(self._gamma, name="gamma")
    # self._momentum_tensor = ops.convert_to_tensor(self._momentum,
    #                                               name="momentum")
    self._epsilon_tensor = ops.convert_to_tensor(self._epsilon,
                                                 name="epsilon")

  def _apply_dense(self, grad, var):
    sd = self.get_slot(var, "sd")
    mom = self.get_slot(var, "mom")
    # if self._centered:
    #   mg = self.get_slot(var, "mg")
    #   return training_ops.apply_centered_sd_prop(
    #       var,
    #       mg,
    #       sd,
    #       mom,
    #       math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
    #       math_ops.cast(self._decay_tensor, var.dtype.base_dtype),
    #       math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
    #       math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype),
    #       grad,
    #       use_locking=self._use_locking).op
    # else:
    return training_ops.apply_sd_prop(
        var,
        sd,
        mom,
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        math_ops.cast(self._gamma_tensor, var.dtype.base_dtype),
        # math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
        math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype),
        grad,
        use_locking=self._use_locking).op

  def _resource_apply_dense(self, grad, var):
    sd = self.get_slot(var, "sd")
    mom = self.get_slot(var, "mom")
    # if self._centered:
    #   mg = self.get_slot(var, "mg")
    #   return training_ops.resource_apply_centered_sd_prop(
    #       var.handle,
    #       mg.handle,
    #       sd.handle,
    #       mom.handle,
    #       math_ops.cast(self._learning_rate_tensor, grad.dtype.base_dtype),
    #       math_ops.cast(self._decay_tensor, grad.dtype.base_dtype),
    #       math_ops.cast(self._momentum_tensor, grad.dtype.base_dtype),
    #       math_ops.cast(self._epsilon_tensor, grad.dtype.base_dtype),
    #       grad,
    #       use_locking=self._use_locking)
    # else:
    return training_ops.resource_apply_sd_prop(
        var.handle,
        sd.handle,
        mom.handle,
        math_ops.cast(self._learning_rate_tensor, grad.dtype.base_dtype),
        math_ops.cast(self._gamma_tensor, grad.dtype.base_dtype),
        # math_ops.cast(self._momentum_tensor, grad.dtype.base_dtype),
        math_ops.cast(self._epsilon_tensor, grad.dtype.base_dtype),
        grad,
        use_locking=self._use_locking)

  def _apply_sparse(self, grad, var):
    sd = self.get_slot(var, "sd")
    mom = self.get_slot(var, "mom")
    # if self._centered:
    #   mg = self.get_slot(var, "mg")
    #   return training_ops.sparse_apply_centered_sd_prop(
    #       var,
    #       mg,
    #       sd,
    #       mom,
    #       math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
    #       math_ops.cast(self._decay_tensor, var.dtype.base_dtype),
    #       math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
    #       math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype),
    #       grad.values,
    #       grad.indices,
    #       use_locking=self._use_locking)
    # else:
    return training_ops.sparse_apply_sd_prop(
        var,
        sd,
        mom,
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        math_ops.cast(self._gamma_tensor, var.dtype.base_dtype),
        # math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
        math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype),
        grad.values,
        grad.indices,
        use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices):
    sd = self.get_slot(var, "sd")
    mom = self.get_slot(var, "mom")
    # if self._centered:
    #   mg = self.get_slot(var, "mg")
    #   return training_ops.resource_sparse_apply_centered_sd_prop(
    #       var.handle,
    #       mg.handle,
    #       sd.handle,
    #       mom.handle,
    #       math_ops.cast(self._learning_rate_tensor, grad.dtype),
    #       math_ops.cast(self._decay_tensor, grad.dtype),
    #       math_ops.cast(self._momentum_tensor, grad.dtype),
    #       math_ops.cast(self._epsilon_tensor, grad.dtype),
    #       grad,
    #       indices,
    #       use_locking=self._use_locking)
    # else:
    return training_ops.resource_sparse_apply_sd_prop(
        var.handle,
        sd.handle,
        mom.handle,
        math_ops.cast(self._learning_rate_tensor, grad.dtype),
        math_ops.cast(self._gamma_tensor, grad.dtype),
        # math_ops.cast(self._momentum_tensor, grad.dtype),
        math_ops.cast(self._epsilon_tensor, grad.dtype),
        grad,
        indices,
        use_locking=self._use_locking)
