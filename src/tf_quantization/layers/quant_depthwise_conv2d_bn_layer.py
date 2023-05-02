# Project: Bachelor Thesis: Automated Quantization of Neural Networks
# Author: Miroslav Safar (xsafar23@stud.fit.vutbr.cz)

import tensorflow as tf

from keras import backend
from keras.utils import control_flow_util

from tf_quantization.layers.base.quant_fused_depthwise_conv2D_batch_norm_layer_base import \
    QuantFusedDepthwiseConv2DBatchNormalizationLayerBase


class QuantFusedDepthwiseConv2DBatchNormalizationLayer(QuantFusedDepthwiseConv2DBatchNormalizationLayerBase):

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

        if per_channel:
            raise ValueError("This scheme supports only per layer quantization")

    def call(self, inputs, training=None, **kwargs):
        input_shape = inputs.shape

        if training is None:
            training = tf.keras.backend.learning_phase()

        if not training or self.is_frozen():
            moving_std_dev = tf.sqrt(self.moving_variance + self.epsilon)
            folded_weights = self._get_folded_weights(std_dev=moving_std_dev)

            # quantization of weights
            if self.weights_quantizer is not None:
                # For per-channel quantization it must be there, but this scheme makes sense only with per-layer
                # channellast_folded_weights = tf.transpose(folded_weights, [0, 1, 3, 2])
                folded_weights = self.weights_quantizer.__call__(folded_weights, training,
                                                                 weights=self._quantizer_weights)
                # folded_weights = tf.transpose(channellast_folded_weights, [0, 1, 3, 2])

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

        # batch_mean = self.moving_mean
        # batch_variance = self.moving_variance

        moving_std_dev = tf.sqrt(self.moving_variance + self.epsilon)
        folded_weights = self._get_folded_weights(std_dev=moving_std_dev)

        # quantization of weights
        if self.weights_quantizer is not None:
            # channellast_folded_weights = tf.transpose(folded_weights, [0, 1, 3, 2])
            folded_weights = self.weights_quantizer.__call__(folded_weights, training,
                                                             weights=self._quantizer_weights)
            # folded_weights = tf.transpose(channellast_folded_weights, [0, 1, 3, 2])

        outputs = backend.depthwise_conv2d(
            inputs,
            folded_weights,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format,
        )

        # Pro použití předchozího schéma stačí zakomentovat tyhle řádky a u get_folded_weights dát batch_variance
        batch_std_dev = tf.math.sqrt(batch_variance + self.epsilon)
        outputs = outputs * (moving_std_dev / batch_std_dev)

        if self.use_bias:
            outputs = self._add_folded_bias(outputs, self.bias, batch_mean, batch_std_dev)
        else:
            outputs = self._add_folded_bias(outputs, [0], batch_mean, batch_std_dev)

        return outputs
