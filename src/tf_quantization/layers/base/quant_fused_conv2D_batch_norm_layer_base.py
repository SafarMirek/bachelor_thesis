# Project: Bachelor Thesis: Automated Quantization of Neural Networks
# Author: Miroslav Safar (xsafar23@stud.fit.vutbr.cz)
import abc

import tensorflow as tf
from keras import initializers, regularizers, constraints, backend
from keras.utils import tf_utils, control_flow_util
from tensorflow import keras
from tensorflow_model_optimization.python.core.quantization.keras import quantizers

from tf_quantization.quantize_registry import DisableableMovingAverageQuantizer


class QuantFusedConv2DBatchNormalizationLayerBase(keras.layers.Conv2D):
    """
    Base class for implementation of methods for solving problem with batch normalization folding during QAT
    for Conv2D + Batch Normalization
    """

    def __init__(self, filters, kernel_size, strides, padding, data_format, dilation_rate, groups, use_bias,
                 kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, kernel_constraint,
                 bias_constraint, axis, momentum, epsilon, center, scale, beta_initializer,
                 gamma_initializer, moving_mean_initializer, moving_variance_initializer, beta_regularizer,
                 gamma_regularizer, beta_constraint, gamma_constraint, quantize, quantize_num_bits_weight,
                 per_channel, symmetric, quantize_outputs, quantize_num_bits_output, **kwargs):
        super().__init__(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                         data_format=data_format, dilation_rate=dilation_rate, groups=groups, use_bias=use_bias,
                         kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint, **kwargs)

        # BatchNormalization params
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(
            moving_variance_initializer
        )
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.quantize = quantize
        self.quantize_outputs = quantize_outputs
        self.quantize_num_bits_weight = quantize_num_bits_weight
        self.quantize_num_bits_output = quantize_num_bits_output

        # Added param that allows batch norm freezing at the end of training
        self.frozen_bn = False

        # Quantization params
        self.per_channel = per_channel
        self.symmetric = symmetric

        self._quantizer_weights = None
        if quantize:
            self.weights_quantizer = quantizers.LastValueQuantizer(
                num_bits=quantize_num_bits_weight,
                per_axis=self.per_channel,
                symmetric=self.symmetric,
                narrow_range=True
            )
        else:
            self.weights_quantizer = None

        self._output_quantizer_vars = None
        if quantize_outputs:
            self.output_quantizer = DisableableMovingAverageQuantizer(
                num_bits=quantize_num_bits_output, per_axis=False,
                symmetric=False, narrow_range=False, min_initializer=keras.initializers.Constant(-6.0),
                max_initializer=keras.initializers.Constant(6.0), no_affect=False)
        else:
            self.output_quantizer = None

    def build(self, input_shape):
        """
        From: https://github.com/keras-team/keras
        """
        super().build(input_shape)
        self.axis = tf_utils.validate_axis(self.axis, input_shape)

        # self.compute_output_shape contains validations for input_shape
        conv_output_shape = self.compute_output_shape(input_shape)

        axis_to_dim = {x: conv_output_shape.dims[x].value for x in self.axis}
        for x in axis_to_dim:
            if axis_to_dim[x] is None:
                raise ValueError(
                    "Input has undefined `axis` dimension. Received input "
                    f"with shape {tuple(conv_output_shape)} "
                    f"and axis={tuple(self.axis)}"
                )

        if len(axis_to_dim) == 1:
            # Single axis batch norm (most common/default use-case)
            param_shape = (list(axis_to_dim.values())[0],)
        else:
            # Parameter shape is the original shape but with 1 in all non-axis
            # dims
            param_shape = [
                axis_to_dim[i] if i in axis_to_dim else 1 for i in range(self.conv_output_shape.rank)
            ]
        self._param_shape = param_shape
        if self.scale:
            self.gamma = self.add_weight(
                name="gamma",
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
                experimental_autocast=False,
            )
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(
                name="beta",
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
                experimental_autocast=False,
            )
        else:
            self.beta = None

        try:
            # Disable variable partitioning when creating the moving mean and
            # variance
            if hasattr(self, "_scope") and self._scope:
                partitioner = self._scope.partitioner
                self._scope.set_partitioner(None)
            else:
                partitioner = None
            self.moving_mean = self.add_weight(
                name="moving_mean",
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.moving_mean_initializer,
                synchronization=tf.VariableSynchronization.ON_READ,
                trainable=False,
                aggregation=tf.VariableAggregation.MEAN,
                experimental_autocast=False,
            )

            self.moving_variance = self.add_weight(
                name="moving_variance",
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.moving_variance_initializer,
                synchronization=tf.VariableSynchronization.ON_READ,
                trainable=False,
                aggregation=tf.VariableAggregation.MEAN,
                experimental_autocast=False,
            )
        finally:
            if partitioner:
                self._scope.set_partitioner(partitioner)

        self._build_quantizer_weights()

        if self.output_quantizer is not None:
            self._output_quantizer_vars = self.output_quantizer.build(
                self.compute_output_shape(input_shape), 'output', self)

        self.built = True

    def get_config(self):
        base_config = super().get_config()
        config = {
            "axis": self.axis,
            "momentum": self.momentum,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": initializers.serialize(self.beta_initializer),
            "gamma_initializer": initializers.serialize(self.gamma_initializer),
            "moving_mean_initializer": initializers.serialize(
                self.moving_mean_initializer
            ),
            "moving_variance_initializer": initializers.serialize(
                self.moving_variance_initializer
            ),
            "beta_regularizer": regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": regularizers.serialize(self.gamma_regularizer),
            "beta_constraint": constraints.serialize(self.beta_constraint),
            "gamma_constraint": constraints.serialize(self.gamma_constraint),
            "quantize": self.quantize,
            "quantize_num_bits_weight": self.quantize_num_bits_weight,
            "per_channel": self.per_channel,
            "symmetric": self.symmetric,
            "quantize_outputs": self.quantize_outputs,
            "quantize_num_bits_output": self.quantize_num_bits_output
        }
        return dict(list(base_config.items()) + list(config.items()))

    def _get_folded_weights(self, std_dev, kernel):
        """
        Folds batch normalization parameters into kernel

        kernel_fold = ( gamma / std_dev) * kernel

        :param std_dev: Standard deviation
        :param kernel: Kernel
        :return: Folded kernel
        """
        gamma = tf.reshape(self.gamma, (1, 1, 1, self.gamma.shape[0]))
        std_dev = tf.reshape(std_dev, (1, 1, 1, std_dev.shape[0]))
        return (gamma / std_dev) * kernel

    def _add_folded_bias(self, outputs, bias, mean, std_dev):
        """
        Folds batch normalization parameters into bias and adds it to output values

        bias_fold = ( gamma / std_dev) * (bias - mean) + beta

        :param outputs: Outputs
        :param bias: Original bias
        :param mean: Mean
        :param std_dev: Standard deviation
        :return: Outputs with added folded bias
        """
        bias = (bias - mean) * (
                self.gamma / std_dev) + self.beta
        return tf.nn.bias_add(
            outputs, bias, data_format=self._tf_data_format
        )

    def _assign_moving_average(self, variable, value, momentum, inputs_size):
        """
        From: https://github.com/keras-team/keras
        """

        def calculate_update_delta():
            decay = tf.convert_to_tensor(1.0 - momentum, name="decay")
            if decay.dtype != variable.dtype.base_dtype:
                decay = tf.cast(decay, variable.dtype.base_dtype)
            update_delta = (variable - tf.cast(value, variable.dtype)) * decay
            if inputs_size is not None:
                update_delta = tf.where(
                    inputs_size > 0,
                    update_delta,
                    backend.zeros_like(update_delta),
                )
            return update_delta

        with backend.name_scope("AssignMovingAvg") as scope:
            if tf.compat.v1.executing_eagerly_outside_functions():
                return variable.assign_sub(calculate_update_delta(), name=scope)
            else:
                with tf.compat.v1.colocate_with(variable):
                    return tf.compat.v1.assign_sub(
                        variable, calculate_update_delta(), name=scope
                    )

    def _assign_new_value(self, variable, value):
        """
        From: https://github.com/keras-team/keras
        """
        with backend.name_scope("AssignNewValue") as scope:
            if tf.compat.v1.executing_eagerly_outside_functions():
                return variable.assign(value, name=scope)
            else:
                with tf.compat.v1.colocate_with(variable):
                    return tf.compat.v1.assign(variable, value, name=scope)

    def _build_quantizer_weights(self):
        if self.weights_quantizer is not None:
            self._quantizer_weights = self.weights_quantizer.build(self.kernel.shape, "weights", self)

    @property
    def _param_dtype(self):
        """
        From: https://github.com/keras-team/keras
        """
        # Raise parameters of fp16 batch norm to fp32
        if self.dtype == tf.float16 or self.dtype == tf.bfloat16:
            return tf.float32
        else:
            return self.dtype or tf.float32

    def freeze_bn(self):
        """
        Freezes BatchNorm in the layer (moving mean and variance won't be updated anymore)
        and training will use moving mean and variance instead of batch statistics

        Graph will be same as inference graph
        """
        self.frozen_bn = True

    def is_frozen(self):
        """
        Returns if batch normalization parameters are frozen
        :return: True if batch norm parameters are frozen
        """
        return self.frozen_bn

    def _apply_quantizer_if_defined(self, *, training, folded_weights):
        """
        Quantize weights if quantizer is defined
        :return: Quantized weights
        """
        if self.weights_quantizer is not None:
            folded_weights = self.weights_quantizer.__call__(folded_weights, training,
                                                             weights=self._quantizer_weights)
        return folded_weights

    def _apply_outputs_quantizer_if_defined(self, *, training, outputs):
        if self.output_quantizer is None:
            return outputs

        return control_flow_util.smart_cond(
            training, self._make_quantizer_fn(self.output_quantizer, outputs, True, self._output_quantizer_vars),
            self._make_quantizer_fn(self.output_quantizer, outputs, False, self._output_quantizer_vars)
        )

    def _make_quantizer_fn(self, quantizer, x, training, quantizer_vars):
        """Use currying to return True/False specialized fns to the cond."""

        def quantizer_fn():
            return quantizer(x, training, weights=quantizer_vars)

        return quantizer_fn

    def call(self, inputs, training=None, **kwargs):
        input_shape = inputs.shape

        if training is None:
            training = tf.keras.backend.learning_phase()

        if not training or self.is_frozen():
            return self._call_bn_frozen(inputs, input_shape, training)
        else:
            return self._call_with_bn(inputs, input_shape, training)

    @abc.abstractmethod
    def _call_bn_frozen(self, inputs, input_shape, training):
        """
        Execution graph for validation and training with frozen batch normalization
        """
        pass

    @abc.abstractmethod
    def _call_with_bn(self, inputs, input_shape, training):
        """
        Execution graph for training with not fronzen batch normalozation
        """
        pass
