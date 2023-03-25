import tensorflow as tf
from tensorflow import keras

from keras import activations
from keras import constraints
from keras import initializers
from keras import regularizers

from keras import backend
from keras.utils import conv_utils
from keras.utils import tf_utils
from keras.utils import control_flow_util
from keras.engine.input_spec import InputSpec
from tensorflow_model_optimization.python.core.quantization.keras import quantizers

from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit import \
    default_n_bit_quantizers


class QuantConv2DBatchLayer(keras.layers.Conv2D):

    def __init__(self, filters, kernel_size, strides, padding, data_format, dilation_rate, groups, use_bias,
                 kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, kernel_constraint,
                 bias_constraint, axis, momentum, epsilon, center, scale, beta_initializer,
                 gamma_initializer, moving_mean_initializer, moving_variance_initializer, beta_regularizer,
                 gamma_regularizer, beta_constraint, gamma_constraint, quantize=True, quantize_num_bits_weight=8,
                 **kwargs):
        super().__init__(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                         data_format=data_format, dilation_rate=dilation_rate, groups=groups, use_bias=use_bias,
                         kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint, **kwargs)

        # TODO: This copy only structure, not weights
        # TODO: Copy weights
        # Convolution params
        # TODO: I currently do not support more that 1 groups

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
        self.quantize_num_bits_weight = quantize_num_bits_weight

        self._frozen_bn = False

        # TODO: this is per channel
        self._quantizer_weights = None
        if quantize:
            self.weights_quantizer = quantizers.LastValueQuantizer(
                num_bits=quantize_num_bits_weight,
                per_axis=True,
                symmetric=True,
                narrow_range=True
            )
        else:
            self.weights_quantizer = None

    def build(self, input_shape):
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

        if self.weights_quantizer is not None:
            self._quantizer_weights = self.weights_quantizer.build(self.kernel.shape, "weights", self)

        self.built = True

    def _get_folded_weights(self, std_dev):
        gamma = tf.reshape(self.gamma, (1, 1, 1, self.gamma.shape[0]))
        std_dev = tf.reshape(std_dev, (1, 1, 1, std_dev.shape[0]))
        return (gamma / std_dev) * self.kernel

    def _add_folded_bias(self, outputs, bias, mean, std_dev):
        # TODO: Handle multiple axes batch normalization
        bias = (bias - mean) * (
                self.gamma / std_dev) + self.beta
        return tf.nn.bias_add(
            outputs, bias, data_format=self._tf_data_format
        )

    def call(self, inputs, training=None, **kwargs):
        input_shape = inputs.shape

        if training is None:
            training = tf.keras.backend.learning_phase()

        if self._is_causal:  # Apply causal padding to inputs for Conv1D.
            inputs = tf.pad(inputs, self._compute_causal_padding(inputs))

        if not training or self._frozen_bn:
            moving_std_dev = tf.sqrt(self.moving_variance + self.epsilon)
            folded_weights = self._get_folded_weights(std_dev=moving_std_dev)

            # quantization of weights
            if self.weights_quantizer is not None:
                folded_weights = self.weights_quantizer.__call__(folded_weights, training,
                                                                 weights=self._quantizer_weights)

            outputs = self.convolution_op(inputs, folded_weights)
            if self.use_bias:
                outputs = self._add_folded_bias(outputs, self.bias, self.moving_mean, moving_std_dev)
            else:
                outputs = self._add_folded_bias(outputs, [0], self.moving_mean, moving_std_dev)

            if not tf.executing_eagerly() and input_shape.rank:
                # Infer the static output shape:
                out_shape = self.compute_output_shape(input_shape)
                outputs.set_shape(out_shape)

            return outputs

        conv_out = self.convolution_op(inputs, self.kernel)
        if self.use_bias:
            conv_out = tf.nn.bias_add(
                conv_out, self.bias, data_format=self._tf_data_format
            )
        bn_input_shape = conv_out.shape
        ndims = len(bn_input_shape)
        reduction_axes = [i for i in range(ndims) if i not in self.axis]
        batch_mean, batch_variance = tf.nn.moments(conv_out, reduction_axes, keepdims=len(self.axis) > 1)

        # Update moving mean and variance
        new_mean, new_variance = batch_mean, batch_variance

        def _do_update(var, value):
            """Compute the updates for mean and variance."""
            return self._assign_moving_average(
                var, value, self.momentum, input_shape[0]
            )

        def mean_update():
            true_branch = lambda: _do_update(self.moving_mean, new_mean)
            false_branch = lambda: self.moving_mean
            return control_flow_util.smart_cond(
                training, true_branch, false_branch
            )

        def variance_update():
            """Update the moving variance."""
            true_branch = lambda: _do_update(
                self.moving_variance, new_variance
            )

            false_branch = lambda: self.moving_variance
            return control_flow_util.smart_cond(
                training, true_branch, false_branch
            )

        self.add_update(mean_update)
        self.add_update(variance_update)

        # batch_mean = self.moving_mean
        # batch_variance = self.moving_variance

        moving_std_dev = tf.sqrt(self.moving_variance + self.epsilon)
        folded_weights = self._get_folded_weights(std_dev=moving_std_dev)

        # quantization of weights
        if self.weights_quantizer is not None:
            folded_weights = self.weights_quantizer.__call__(folded_weights, training,
                                                             weights=self._quantizer_weights)

        outputs = self.convolution_op(inputs, folded_weights)

        # Pro použití předchozího schéma stačí zakomentovat tyhle řádky a u get_folded_weights dát batch_variance
        batch_std_dev = tf.math.sqrt(batch_variance + self.epsilon)
        outputs = outputs * (moving_std_dev / batch_std_dev)

        if self.use_bias:
            outputs = self._add_folded_bias(outputs, self.bias, batch_mean, batch_std_dev)
        else:
            outputs = self._add_folded_bias(outputs, [0], batch_mean, batch_std_dev)

        return outputs

    def _assign_moving_average(self, variable, value, momentum, inputs_size):
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
        with backend.name_scope("AssignNewValue") as scope:
            if tf.compat.v1.executing_eagerly_outside_functions():
                return variable.assign(value, name=scope)
            else:
                with tf.compat.v1.colocate_with(variable):
                    return tf.compat.v1.assign(variable, value, name=scope)

    @property
    def _param_dtype(self):
        # Raise parameters of fp16 batch norm to fp32
        if self.dtype == tf.float16 or self.dtype == tf.bfloat16:
            return tf.float32
        else:
            return self.dtype or tf.float32

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
            "quantize_num_bits_weight": self.quantize_num_bits_weight
        }
        return dict(list(base_config.items()) + list(config.items()))

    def freeze_bn(self):
        """
        Freezes BatchNorm in the layer (moving mean and variance won't be updated anymore)
        and training will use moving mean and variance instead of batch statistics

        Graph will be same as inference graph
        """
        self._frozen_bn = True
