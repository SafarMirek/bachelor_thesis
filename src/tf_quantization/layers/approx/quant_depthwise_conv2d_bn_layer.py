import tensorflow as tf
from tensorflow import keras

from keras import constraints
from keras import initializers
from keras import regularizers

from keras import backend
from keras.utils import tf_utils
from keras.utils import control_flow_util
from tensorflow_model_optimization.python.core.quantization.keras import quantizers

from tf_quantization.layers.base.quant_fused_depthwise_conv2D_batch_norm_layer_base import \
    QuantFusedDepthwiseConv2DBatchNormalizationLayerBase


class ApproxQuantFusedDepthwiseConv2DBatchNormalizationLayer(QuantFusedDepthwiseConv2DBatchNormalizationLayerBase):

    def __init__(self, kernel_size, strides, padding, depth_multiplier, data_format, dilation_rate, activation,
                 use_bias, depthwise_initializer, bias_initializer, depthwise_regularizer, bias_regularizer,
                 activity_regularizer, depthwise_constraint, bias_constraint,
                 axis, momentum, epsilon, center, scale, beta_initializer,
                 gamma_initializer, moving_mean_initializer, moving_variance_initializer, beta_regularizer,
                 gamma_regularizer, beta_constraint, gamma_constraint, quantize, quantize_num_bits_weight,
                 per_channel, symmetric, **kwargs):
        super().__init__(kernel_size=kernel_size, strides=strides, padding=padding, depth_multiplier=depth_multiplier,
                         data_format=data_format,
                         dilation_rate=dilation_rate,
                         activation=activation,
                         use_bias=use_bias, depthwise_initializer=depthwise_initializer,
                         bias_initializer=bias_initializer,
                         depthwise_regularizer=depthwise_regularizer,
                         bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer, depthwise_constraint=depthwise_constraint,
                         bias_constraint=bias_constraint,
                         axis=axis, momentum=momentum, epsilon=epsilon, center=center, scale=scale,
                         beta_initializer=beta_initializer,
                         gamma_initializer=gamma_initializer, moving_mean_initializer=moving_mean_initializer,
                         moving_variance_initializer=moving_variance_initializer, beta_regularizer=beta_regularizer,
                         gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint,
                         gamma_constraint=gamma_constraint, quantize=quantize,
                         quantize_num_bits_weight=quantize_num_bits_weight,
                         per_channel=per_channel, symmetric=symmetric
                         , **kwargs)

    def build(self, input_shape):
        super().build(self, input_shape)

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

        self._build_quantizer_weights()

        self.built = True

    def _reset_folded_weights(self, std_dev, outputs):
        gamma = tf.reshape(self.gamma, (1, 1, 1, self.gamma.shape[0]))
        std_dev = tf.reshape(std_dev, (1, 1, 1, std_dev.shape[0]))
        return (std_dev / gamma) * outputs

    def call(self, inputs, training=None, **kwargs):
        input_shape = inputs.shape

        if training is None:
            training = tf.keras.backend.learning_phase()

        if not training:
            moving_std_dev = tf.math.sqrt(self.moving_variance + self.epsilon)
            if not self.per_channel:
                folded_weights = self._get_folded_weights(std_dev=moving_std_dev,
                                                          depthwise_kernel=self.depthwise_kernel)
                if self.weights_quantizer is not None:
                    folded_weights = self.weights_quantizer.__call__(folded_weights, training,
                                                                     weights=self._quantizer_weights)
            else:
                folded_weights = self.depthwise_kernel
                # quantization of weights
                if self.weights_quantizer is not None:
                    folded_weights = tf.transpose(folded_weights, [0, 1, 3, 2])
                    folded_weights = self.weights_quantizer.__call__(folded_weights, training,
                                                                     weights=self._quantizer_weights)
                    folded_weights = tf.transpose(folded_weights, [0, 1, 3, 2])

                folded_weights = self._get_folded_weights(std_dev=moving_std_dev,
                                                          depthwise_kernel=folded_weights)

            outputs = backend.depthwise_conv2d(
                inputs,
                folded_weights,
                strides=self.strides,
                padding=self.padding,
                dilation_rate=self.dilation_rate,
                data_format=self.data_format,
            )

            if self.use_bias:
                outputs = self._add_folded_bias(outputs, self.bias, self.moving_mean, moving_std_dev)
            else:
                outputs = self._add_folded_bias(outputs, [0], self.moving_mean, moving_std_dev)

            return outputs

        moving_std_dev = tf.math.sqrt(self.moving_variance + self.epsilon)
        if not self.per_channel:
            folded_weights = self._get_folded_weights(std_dev=moving_std_dev, depthwise_kernel=self.depthwise_kernel)
        else:
            folded_weights = self.depthwise_kernel

        # quantization of weights
        if self.weights_quantizer is not None:
            if self.per_channel:
                folded_weights = tf.transpose(folded_weights, [0, 1, 3, 2])
            folded_weights = self.weights_quantizer.__call__(folded_weights, training,
                                                             weights=self._quantizer_weights)
            if self.per_channel:
                folded_weights = tf.transpose(folded_weights, [0, 1, 3, 2])

        if self.is_frozen() and self.per_channel:
            # If bn is frozen we need to fold weights, since ranges for per-channel quantization are not
            # scaled we need to scale weights after quantization and not before it
            folded_weights = self._get_folded_weights(std_dev=moving_std_dev, depthwise_kernel=folded_weights)

        outputs = backend.depthwise_conv2d(
            inputs,
            folded_weights,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format,
        )

        if self.is_frozen():
            if self.use_bias:
                outputs = self._add_folded_bias(outputs, self.bias, self.moving_mean, moving_std_dev)
            else:
                outputs = self._add_folded_bias(outputs, [0], self.moving_mean, moving_std_dev)
            return outputs

        # * ( std_dev / gamma)
        if not self.per_channel:
            outputs = self._reset_folded_weights(moving_std_dev, outputs)

        if self.use_bias:
            outputs = backend.bias_add(
                outputs, self.bias, data_format=self._tf_data_format
            )

        bn_input_shape = outputs.shape
        ndims = len(bn_input_shape)
        reduction_axes = [i for i in range(ndims) if i not in self.axis]
        batch_mean, batch_variance = tf.nn.moments(outputs, reduction_axes, keepdims=len(self.axis) > 1)

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

        broadcast_shape = [1] * ndims
        broadcast_shape[self.axis[0]] = input_shape.dims[self.axis[0]].value

        def _broadcast(v):
            if (
                    v is not None
                    and len(v.shape) != ndims
                    and reduction_axes != list(range(ndims - 1))
            ):
                return tf.reshape(v, broadcast_shape)
            return v

        outputs = tf.nn.batch_normalization(
            outputs,
            _broadcast(batch_mean),
            _broadcast(batch_variance),
            _broadcast(self.beta),
            _broadcast(self.gamma),
            self.epsilon,
        )

        return outputs
