import tensorflow as tf

from keras.utils import control_flow_util

from tf_quantization.layers.base.quant_fused_conv2D_batch_norm_layer_base import \
    QuantFusedConv2DBatchNormalizationLayerBase


class QuantFusedConv2DBatchNormalizationLayer(QuantFusedConv2DBatchNormalizationLayerBase):

    def __init__(self, filters, kernel_size, strides, padding, data_format, dilation_rate, groups, use_bias,
                 kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, kernel_constraint,
                 bias_constraint, axis, momentum, epsilon, center, scale, beta_initializer,
                 gamma_initializer, moving_mean_initializer, moving_variance_initializer, beta_regularizer,
                 gamma_regularizer, beta_constraint, gamma_constraint, quantize, quantize_num_bits_weight,
                 per_channel, symmetric, **kwargs):
        super().__init__(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                         data_format=data_format, dilation_rate=dilation_rate, groups=groups, use_bias=use_bias,
                         kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint, axis=axis, momentum=momentum, epsilon=epsilon, center=center,
                         scale=scale, beta_initializer=beta_initializer,
                         gamma_initializer=gamma_initializer, moving_mean_initializer=moving_mean_initializer,
                         moving_variance_initializer=moving_variance_initializer, beta_regularizer=beta_regularizer,
                         gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint,
                         gamma_constraint=gamma_constraint, quantize=quantize,
                         quantize_num_bits_weight=quantize_num_bits_weight,
                         per_channel=per_channel, symmetric=symmetric, **kwargs)

        if per_channel:
            raise ValueError("This scheme supports only per layer quantization")

    def call(self, inputs, training=None, **kwargs):
        input_shape = inputs.shape

        if training is None:
            training = tf.keras.backend.learning_phase()

        if self._is_causal:  # Apply causal padding to inputs for Conv1D.
            inputs = tf.pad(inputs, self._compute_causal_padding(inputs))

        if not training or self.is_frozen():
            moving_std_dev = tf.sqrt(self.moving_variance + self.epsilon)
            folded_weights = self._get_folded_weights(std_dev=moving_std_dev, kernel=self.kernel)

            # quantization of weights
            folded_weights = self._apply_quantizer_if_defined(training=training, folded_weights=folded_weights)

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

        moving_std_dev = tf.sqrt(self.moving_variance + self.epsilon)
        folded_weights = self._get_folded_weights(std_dev=moving_std_dev, kernel=self.kernel)

        # quantization of weights
        folded_weights = self._apply_quantizer_if_defined(training=training, folded_weights=folded_weights)

        outputs = self.convolution_op(inputs, folded_weights)

        batch_std_dev = tf.math.sqrt(batch_variance + self.epsilon)
        outputs = outputs * (moving_std_dev / batch_std_dev)

        if self.use_bias:
            outputs = self._add_folded_bias(outputs, self.bias, batch_mean, batch_std_dev)
        else:
            outputs = self._add_folded_bias(outputs, [0], batch_mean, batch_std_dev)

        return outputs

    def _apply_quantizer_if_defined(self, *, training, folded_weights):
        if self.weights_quantizer is not None:
            folded_weights = self.weights_quantizer.__call__(folded_weights, training,
                                                             weights=self._quantizer_weights)
        return folded_weights
