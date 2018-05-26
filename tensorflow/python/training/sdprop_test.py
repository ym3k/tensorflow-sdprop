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
"""Tests for sdprop."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import itertools
import math

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import sdprop

_DATA_TYPES = [dtypes.half, dtypes.float32]

_TEST_PARAM_VALUES = [
    # learning_rate, gamma, epsilon, use_resource
    [0.5, 0.9, 1e-3, False],
    [0.5, 0.9, 1e-3, False],
    [0.5, 0.9, 1e-3, True],
    [0.5, 0.9, 1e-3, True],
    [0.1, 0.9, 1e-3, False],
    [0.5, 0.95, 1e-3, False],
    [0.5, 0.95, 1e-5, False],
    [0.5, 0.95, 1e-5, False],
]

_TESTPARAMS = [
    [data_type] + values
    for data_type, values in itertools.product(_DATA_TYPES, _TEST_PARAM_VALUES)
]


class SDPropOptimizerTest(test.TestCase):

  def _sdprop_update_numpy(self, var, g, mg, sd, mom, lr, gamma,
                            epsilon):
    sd_t = sd * decay + (1 - decay) * g * g
    denom_t = sd_t + epsilon
    mg_t = mg
    mom_t = momentum * mom + lr * g / np.sqrt(denom_t, dtype=denom_t.dtype)
    var_t = var - mom_t
    return var_t, mg_t, sd_t, mom_t

  def _sparse_sdprop_update_numpy(self, var, gindexs, gvalues, mg, sd, mom,
                                   lr, gamma, epsilon):
    mg_t = copy.deepcopy(mg)
    sd_t = copy.deepcopy(sd)
    mom_t = copy.deepcopy(mom)
    var_t = copy.deepcopy(var)
    for i in range(len(gindexs)):
      gindex = gindexs[i]
      gvalue = gvalues[i]
      sd_t[gindex] = sd[gindex] * decay + (1 - decay) * gvalue * gvalue
      denom_t = sd_t[gindex] + epsilon
      mom_t[gindex] = momentum * mom[gindex] + lr * gvalue / np.sqrt(denom_t)
      var_t[gindex] = var[gindex] - mom_t[gindex]
    return var_t, mg_t, sd_t, mom_t

  def testDense(self):
    # TODO(yori): Use ParameterizedTest when available
    for (dtype, learning_rate, gamma, 
         epsilon, use_resource) in _TESTPARAMS:
      with self.test_session(use_gpu=True):
        # Initialize variables for numpy implementation.
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.2], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.2], dtype=dtype.as_numpy_dtype)

        if use_resource:
          var0 = resource_variable_ops.ResourceVariable(var0_np)
          var1 = resource_variable_ops.ResourceVariable(var1_np)
        else:
          var0 = variables.Variable(var0_np)
          var1 = variables.Variable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)
        opt = sdprop.SDPropOptimizer(
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon)

        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        mg0 = opt.get_slot(var0, "mg")
        self.assertEqual(mg0 is not None, centered)
        mg1 = opt.get_slot(var1, "mg")
        self.assertEqual(mg1 is not None, centered)
        sd0 = opt.get_slot(var0, "sd")
        self.assertTrue(sd0 is not None)
        sd1 = opt.get_slot(var1, "sd")
        self.assertTrue(sd1 is not None)
        # mom0 = opt.get_slot(var0, "momentum")
        # self.assertTrue(mom0 is not None)
        # mom1 = opt.get_slot(var1, "momentum")
        # self.assertTrue(mom1 is not None)

        mg0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        mg1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        sd0_np = np.array([1.0, 1.0], dtype=dtype.as_numpy_dtype)
        sd1_np = np.array([1.0, 1.0], dtype=dtype.as_numpy_dtype)
        # mom0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        # mom1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())

        # Run 4 steps of SDProp
        for t in range(1, 5):
          update.run()

          var0_np, mg0_np, sd0_np, mom0_np = self._sdprop_update_numpy(
              var0_np, grads0_np, mg0_np, sd0_np, mom0_np, learning_rate,
              decay, momentum, epsilon, centered)
          var1_np, mg1_np, sd1_np, mom1_np = self._sdprop_update_numpy(
              var1_np, grads1_np, mg1_np, sd1_np, mom1_np, learning_rate,
              decay, momentum, epsilon, centered)

          # Validate updated params
          if centered:
            self.assertAllCloseAccordingToType(mg0_np, mg0.eval())
            self.assertAllCloseAccordingToType(mg1_np, mg1.eval())
          self.assertAllCloseAccordingToType(sd0_np, sd0.eval())
          self.assertAllCloseAccordingToType(sd1_np, sd1.eval())
          self.assertAllCloseAccordingToType(mom0_np, mom0.eval())
          self.assertAllCloseAccordingToType(mom1_np, mom1.eval())
          self.assertAllCloseAccordingToType(var0_np, var0.eval())
          self.assertAllCloseAccordingToType(var1_np, var1.eval())

  def testMinimizeSparseResourceVariable(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      with self.test_session():
        var0 = resource_variable_ops.ResourceVariable([[1.0, 2.0]], dtype=dtype)
        x = constant_op.constant([[4.0], [5.0]], dtype=dtype)
        pred = math_ops.matmul(embedding_ops.embedding_lookup([var0], [0]), x)
        loss = pred * pred
        sgd_op = sdprop.SDPropOptimizer(
            learning_rate=1.0,
            decay=0.0,
            momentum=0.0,
            epsilon=0.0,
            centered=False).minimize(loss)
        variables.global_variables_initializer().run()
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([[1.0, 2.0]], var0.eval())
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        self.assertAllCloseAccordingToType(
            [[0., 1.]], var0.eval(), atol=0.01)

  def testMinimizeSparseResourceVariableCentered(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      with self.test_session():
        var0 = resource_variable_ops.ResourceVariable([[1.0, 2.0]], dtype=dtype)
        x = constant_op.constant([[4.0], [5.0]], dtype=dtype)
        pred = math_ops.matmul(embedding_ops.embedding_lookup([var0], [0]), x)
        loss = pred * pred
        sgd_op = sdprop.SDPropOptimizer(
            learning_rate=1.0,
            decay=0.0,
            momentum=0.0,
            epsilon=1.0,
            centered=True).minimize(loss)
        variables.global_variables_initializer().run()
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([[1.0, 2.0]], var0.eval())
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        self.assertAllCloseAccordingToType(
            [[-111, -138]], var0.eval(), atol=0.01)

  def testSparse(self):
    # TODO(yori): Use ParameterizedTest when available
    for (dtype, learning_rate, decay,
         momentum, epsilon, centered, _) in _TESTPARAMS:
      with self.test_session(use_gpu=True):
        # Initialize variables for numpy implementation.
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01], dtype=dtype.as_numpy_dtype)

        var0 = variables.Variable(var0_np)
        var1 = variables.Variable(var1_np)
        grads0_np_indices = np.array([0], dtype=np.int32)
        grads0 = ops.IndexedSlices(
            constant_op.constant(grads0_np),
            constant_op.constant(grads0_np_indices), constant_op.constant([1]))
        grads1_np_indices = np.array([1], dtype=np.int32)
        grads1 = ops.IndexedSlices(
            constant_op.constant(grads1_np),
            constant_op.constant(grads1_np_indices), constant_op.constant([1]))
        opt = sdprop.SDPropOptimizer(
            learning_rate=learning_rate,
            decay=decay,
            momentum=momentum,
            epsilon=epsilon,
            centered=centered)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        mg0 = opt.get_slot(var0, "mg")
        self.assertEqual(mg0 is not None, centered)
        mg1 = opt.get_slot(var1, "mg")
        self.assertEqual(mg1 is not None, centered)
        sd0 = opt.get_slot(var0, "sd")
        self.assertTrue(sd0 is not None)
        sd1 = opt.get_slot(var1, "sd")
        self.assertTrue(sd1 is not None)
        mom0 = opt.get_slot(var0, "momentum")
        self.assertTrue(mom0 is not None)
        mom1 = opt.get_slot(var1, "momentum")
        self.assertTrue(mom1 is not None)

        mg0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        mg1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        sd0_np = np.array([1.0, 1.0], dtype=dtype.as_numpy_dtype)
        sd1_np = np.array([1.0, 1.0], dtype=dtype.as_numpy_dtype)
        mom0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        mom1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())

        # Run 4 steps of SDProp
        for t in range(1, 5):
          update.run()

          var0_np, mg0_np, sd0_np, mom0_np = self._sparse_sdprop_update_numpy(
              var0_np, grads0_np_indices, grads0_np, mg0_np, sd0_np, mom0_np,
              learning_rate, decay, momentum, epsilon, centered)
          var1_np, mg1_np, sd1_np, mom1_np = self._sparse_sdprop_update_numpy(
              var1_np, grads1_np_indices, grads1_np, mg1_np, sd1_np, mom1_np,
              learning_rate, decay, momentum, epsilon, centered)

          # Validate updated params
          if centered:
            self.assertAllCloseAccordingToType(mg0_np, mg0.eval())
            self.assertAllCloseAccordingToType(mg1_np, mg1.eval())
          self.assertAllCloseAccordingToType(sd0_np, sd0.eval())
          self.assertAllCloseAccordingToType(sd1_np, sd1.eval())
          self.assertAllCloseAccordingToType(mom0_np, mom0.eval())
          self.assertAllCloseAccordingToType(mom1_np, mom1.eval())
          self.assertAllCloseAccordingToType(var0_np, var0.eval())
          self.assertAllCloseAccordingToType(var1_np, var1.eval())

  def testWithoutMomentum(self):
    for dtype in [dtypes.half, dtypes.float32]:
      with self.test_session(use_gpu=True):
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        opt = sdprop.SDPropOptimizer(
            learning_rate=2.0, decay=0.9, momentum=0.0, epsilon=1.0)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        sd0 = opt.get_slot(var0, "sd")
        self.assertTrue(sd0 is not None)
        sd1 = opt.get_slot(var1, "sd")
        self.assertTrue(sd1 is not None)
        mom0 = opt.get_slot(var0, "momentum")
        self.assertTrue(mom0 is not None)
        mom1 = opt.get_slot(var1, "momentum")
        self.assertTrue(mom1 is not None)

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())
        # Step 1: the sd accumulators where 1. So we should see a normal
        # update: v -= grad * learning_rate
        update.run()
        # Check the root mean square accumulators.
        self.assertAllCloseAccordingToType(
            np.array([0.901, 0.901]), sd0.eval())
        self.assertAllCloseAccordingToType(
            np.array([0.90001, 0.90001]), sd1.eval())
        # Check the parameters.
        self.assertAllCloseAccordingToType(
            np.array([
                1.0 - (0.1 * 2.0 / math.sqrt(0.901 + 1.0)),
                2.0 - (0.1 * 2.0 / math.sqrt(0.901 + 1.0))
            ]), var0.eval())
        self.assertAllCloseAccordingToType(
            np.array([
                3.0 - (0.01 * 2.0 / math.sqrt(0.90001 + 1.0)),
                4.0 - (0.01 * 2.0 / math.sqrt(0.90001 + 1.0))
            ]), var1.eval())
        # Step 2: the root mean square accumulators contain the previous update.
        update.run()
        # Check the sd accumulators.
        self.assertAllCloseAccordingToType(
            np.array([0.901 * 0.9 + 0.001, 0.901 * 0.9 + 0.001]), sd0.eval())
        self.assertAllCloseAccordingToType(
            np.array([0.90001 * 0.9 + 1e-5, 0.90001 * 0.9 + 1e-5]), sd1.eval())
        # Check the parameters.
        self.assertAllCloseAccordingToType(
            np.array([
                1.0 - (0.1 * 2.0 / math.sqrt(0.901 + 1.0)) -
                (0.1 * 2.0 / math.sqrt(0.901 * 0.9 + 0.001 + 1.0)),
                2.0 - (0.1 * 2.0 / math.sqrt(0.901 + 1.0)) -
                (0.1 * 2.0 / math.sqrt(0.901 * 0.9 + 0.001 + 1.0))
            ]), var0.eval())
        self.assertAllCloseAccordingToType(
            np.array([
                3.0 - (0.01 * 2.0 / math.sqrt(0.90001 + 1.0)) -
                (0.01 * 2.0 / math.sqrt(0.90001 * 0.9 + 1e-5 + 1.0)),
                4.0 - (0.01 * 2.0 / math.sqrt(0.90001 + 1.0)) -
                (0.01 * 2.0 / math.sqrt(0.90001 * 0.9 + 1e-5 + 1.0))
            ]), var1.eval())

  def testWithMomentum(self):
    for dtype in [dtypes.half, dtypes.float32]:
      with self.test_session(use_gpu=True):
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)

        opt = sdprop.SDPropOptimizer(
            learning_rate=2.0, decay=0.9, momentum=0.5, epsilon=1e-5)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        sd0 = opt.get_slot(var0, "sd")
        self.assertTrue(sd0 is not None)
        sd1 = opt.get_slot(var1, "sd")
        self.assertTrue(sd1 is not None)
        mom0 = opt.get_slot(var0, "momentum")
        self.assertTrue(mom0 is not None)
        mom1 = opt.get_slot(var1, "momentum")
        self.assertTrue(mom1 is not None)

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())
        # Step 1: sd = 1, mom = 0. So we should see a normal
        # update: v -= grad * learning_rate
        update.run()
        # Check the root mean square accumulators.
        self.assertAllCloseAccordingToType(
            np.array([0.901, 0.901]), sd0.eval())
        self.assertAllCloseAccordingToType(
            np.array([0.90001, 0.90001]), sd1.eval())
        # Check the momentum accumulators
        self.assertAllCloseAccordingToType(
            np.array([(0.1 * 2.0 / math.sqrt(0.901 + 1e-5)),
                      (0.1 * 2.0 / math.sqrt(0.901 + 1e-5))]), mom0.eval())
        self.assertAllCloseAccordingToType(
            np.array([(0.01 * 2.0 / math.sqrt(0.90001 + 1e-5)),
                      (0.01 * 2.0 / math.sqrt(0.90001 + 1e-5))]), mom1.eval())

        # Check that the parameters.
        self.assertAllCloseAccordingToType(
            np.array([
                1.0 - (0.1 * 2.0 / math.sqrt(0.901 + 1e-5)),
                2.0 - (0.1 * 2.0 / math.sqrt(0.901 + 1e-5))
            ]), var0.eval())
        self.assertAllCloseAccordingToType(
            np.array([
                3.0 - (0.01 * 2.0 / math.sqrt(0.90001 + 1e-5)),
                4.0 - (0.01 * 2.0 / math.sqrt(0.90001 + 1e-5))
            ]), var1.eval())

        # Step 2: the root mean square accumulators contain the previous update.
        update.run()
        # Check the sd accumulators.
        self.assertAllCloseAccordingToType(
            np.array([0.901 * 0.9 + 0.001, 0.901 * 0.9 + 0.001]), sd0.eval())
        self.assertAllCloseAccordingToType(
            np.array([0.90001 * 0.9 + 1e-5, 0.90001 * 0.9 + 1e-5]), sd1.eval())
        self.assertAllCloseAccordingToType(
            np.array([
                0.5 * (0.1 * 2.0 / math.sqrt(0.901 + 1e-5)) +
                (0.1 * 2.0 / math.sqrt(0.901 * 0.9 + 0.001 + 1e-5)),
                0.5 * (0.1 * 2.0 / math.sqrt(0.901 + 1e-5)) +
                (0.1 * 2.0 / math.sqrt(0.901 * 0.9 + 0.001 + 1e-5))
            ]), mom0.eval())
        self.assertAllCloseAccordingToType(
            np.array([
                0.5 * (0.01 * 2.0 / math.sqrt(0.90001 + 1e-5)) +
                (0.01 * 2.0 / math.sqrt(0.90001 * 0.9 + 2e-5)),
                0.5 * (0.01 * 2.0 / math.sqrt(0.90001 + 1e-5)) +
                (0.01 * 2.0 / math.sqrt(0.90001 * 0.9 + 2e-5))
            ]), mom1.eval())

        # Check the parameters.
        self.assertAllCloseAccordingToType(
            np.array([
                1.0 - (0.1 * 2.0 / math.sqrt(0.901 + 1e-5)) -
                (0.5 * (0.1 * 2.0 / math.sqrt(0.901 + 1e-5)) +
                 (0.1 * 2.0 / math.sqrt(0.901 * 0.9 + 0.001 + 1e-5))),
                2.0 - (0.1 * 2.0 / math.sqrt(0.901 + 1e-5)) -
                (0.5 * (0.1 * 2.0 / math.sqrt(0.901 + 1e-5)) +
                 (0.1 * 2.0 / math.sqrt(0.901 * 0.9 + 0.001 + 1e-5)))
            ]), var0.eval())

        self.assertAllCloseAccordingToType(
            np.array([
                3.0 - (0.01 * 2.0 / math.sqrt(0.90001 + 1e-5)) -
                (0.5 * (0.01 * 2.0 / math.sqrt(0.90001 + 1e-5)) +
                 (0.01 * 2.0 / math.sqrt(0.90001 * 0.9 + 2e-5))),
                4.0 - (0.01 * 2.0 / math.sqrt(0.90001 + 1e-5)) -
                (0.5 * (0.01 * 2.0 / math.sqrt(0.90001 + 1e-5)) +
                 (0.01 * 2.0 / math.sqrt(0.90001 * 0.9 + 2e-5)))
            ]), var1.eval())


if __name__ == "__main__":
  test.main()
