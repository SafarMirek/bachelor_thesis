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

from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit import \
    default_n_bit_quantizers


class QuantDepthwiseConv2DBatchNormalizationLayer(keras.layers.DepthwiseConv2D):

    def __init__(self, dephwise_conv_layer: keras.layers.DepthwiseConv2D, bn_layer: keras.layers.BatchNormalization,
                 quantize=True, **kwargs):
        super().__init__(**dephwise_conv_layer.get_config())

        self._dephwise_conv_layer = dephwise_conv_layer
        self._bn_layer = bn_layer

        # Batch Normalization
        self.axis = self._bn_layer.axis
        self.momentum = self._bn_layer.momentum
        self.epsilon = self._bn_layer.epsilon
        self.center = self._bn_layer.center
        self.scale = self._bn_layer.scale
        self.beta_initializer = self._bn_layer.beta_initializer
        self.gamma_initializer = self._bn_layer.gamma_initializer
        self.moving_mean_initializer = self._bn_layer.moving_mean_initializer
        self.moving_variance_initializer = self._bn_layer.moving_variance_initializer
        self.beta_regularizer = self._bn_layer.beta_regularizer
        self.gamma_regularizer = self._bn_layer.gamma_regularizer
        self.beta_constraint = self._bn_layer.beta_constraint
        self.gamma_constraint = self._bn_layer.gamma_constraint

        # TODO: This copy only structure, not weights
        # TODO: Copy weights
        # TODO: I currently do not support more that 1 groups

        # TODO: this is per channel
        if quantize:
            self.weights_quantizer = default_n_bit_quantizers.DefaultNBitConvWeightsQuantizer()
        else:
            self.weights_quantizer = None

    def build(self, input_shape):
        keras.layers.DepthwiseConv2D.build(self, input_shape)

        self.axis = tf_utils.validate_axis(self.axis, input_shape)
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
            self.weights_quantizer.build(input_shape, "weights", self)

        # Copy weights from original convolution and batch norm layer
        conv_weights = self._dephwise_conv_layer.get_weights()
        conv_weights_len = len(conv_weights)
        bn_weights = self._bn_layer.get_weights()
        current_weights = self.get_weights()
        for i in range(conv_weights_len):
            current_weights[i] = conv_weights[i]

        for j in range(len(conv_weights)):
            current_weights[j + conv_weights_len] = bn_weights[j]

        self.built = True

    def _get_folded_weights(self, variance):
        return (self.gamma / tf.math.sqrt(variance + self.epsilon)) * self.depthwise_kernel

    def _add_folded_bias(self, outputs, mean, variance, folded=True):
        # TODO: Handle multiple axes batch normalization
        if folded:
            bias = (self.bias - mean) * (
                    self.gamma / tf.math.sqrt(variance + self.epsilon)) + self.beta
        else:
            bias = self.bias
        return tf.nn.bias_add(
            outputs, bias, data_format=self._tf_data_format
        )

    def call(self, inputs, training=None, **kwargs):
        input_shape = inputs.shape

        if training is None:
            training = tf.keras.backend.learning_phase()

        if not training:
            folded_weights = self._get_folded_weights(variance=self.moving_variance)

            # quantization of weights
            if self.weights_quantizer is not None:
                folded_weights = self.weights_quantizer.__call__(inputs, training, folded_weights)

            outputs = backend.depthwise_conv2d(
                inputs,
                folded_weights,
                strides=self.strides,
                padding=self.padding,
                dilation_rate=self.dilation_rate,
                data_format=self.data_format,
            )

            if self.use_bias:
                outputs = self._add_folded_bias(outputs, self.moving_mean, self.moving_variance)

            # TODO: Activation and activation quantization
            # if self.activation is not None:
            #    return self.activation(outputs)

            return outputs

        conv_out = backend.depthwise_conv2d(
            inputs,
            self.depthwise_kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format,
        )
        if self.use_bias:
            conv_out = backend.bias_add(
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

        folded_weights = self._get_folded_weights(variance=batch_variance)

        # quantization of weights
        if self.weights_quantizer is not None:
            folded_weights = self.weights_quantizer.__call__(inputs, training, folded_weights)

        outputs = backend.depthwise_conv2d(
            inputs,
            folded_weights,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format,
        )

        if self.use_bias:
            outputs = self._add_folded_bias(outputs, batch_mean, batch_variance)

        return outputs

    # TODO: get_config(), set_config()

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
