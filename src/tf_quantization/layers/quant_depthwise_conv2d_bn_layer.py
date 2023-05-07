# Project: Bachelor Thesis: Automated Quantization of Neural Networks
# Author: Miroslav Safar (xsafar23@stud.fit.vutbr.cz)
from abc import ABC

import tensorflow as tf

from keras import backend
from keras.utils import control_flow_util

from tf_quantization.layers.base.quant_fused_depthwise_conv2D_batch_norm_layer_base import \
    QuantFusedDepthwiseConv2DBatchNormalizationLayerBase


class QuantFusedDepthwiseConv2DBatchNormalizationLayer(QuantFusedDepthwiseConv2DBatchNormalizationLayerBase):
    """
    This class implements method for solving problem with batch normalization folding during QAT
    for DepthwiseConv2D + Batch Normalization
    """

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

    def _call__bn_frozen(self, inputs, training):
        """
        Execution graph for validation and training with frozen batch normalization
        """
        moving_std_dev = tf.sqrt(self.moving_variance + self.epsilon)
        folded_weights = self._get_folded_weights(std_dev=moving_std_dev, depthwise_kernel=self.depthwise_kernel)

        # quantization of weights
        folded_weights = self._apply_quantizer_if_defined(folded_weights=folded_weights, training=training)

        outputs = backend.depthwise_conv2d(
            inputs,
            folded_weights,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format,
        )

        outputs = self._add_folded_bias(outputs, self.bias if self.use_bias else [0], self.moving_mean,
                                        moving_std_dev)

        return outputs

    def _call_with_bn(self, inputs, input_shape, training):
        """
        Execution graph for training with not fronzen batch normalozation
        """
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
            """
            Compute the updates for mean and variance.
            From: https://github.com/tensorflow/model-optimization
            """
            return self._assign_moving_average(
                var, value, self.momentum, input_shape[0]
            )

        def mean_update():
            """
            Update the moving mean
            From: https://github.com/tensorflow/model-optimization
            """
            true_branch = lambda: _do_update(self.moving_mean, new_mean)
            false_branch = lambda: self.moving_mean
            return control_flow_util.smart_cond(
                training, true_branch, false_branch
            )

        def variance_update():
            """
            Update the moving variance.
            From: https://github.com/tensorflow/model-optimization
            """
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
        folded_weights = self._get_folded_weights(std_dev=moving_std_dev, depthwise_kernel=self.depthwise_kernel)

        # quantization of weights
        folded_weights = self._apply_quantizer_if_defined(folded_weights=folded_weights, training=training)

        outputs = backend.depthwise_conv2d(
            inputs,
            folded_weights,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format,
        )

        batch_std_dev = tf.math.sqrt(batch_variance + self.epsilon)
        outputs = outputs * (moving_std_dev / batch_std_dev)

        outputs = self._add_folded_bias(outputs, self.bias if self.use_bias else [0], batch_mean, batch_std_dev)

        return outputs

    def call(self, inputs, training=None, **kwargs):
        input_shape = inputs.shape

        if training is None:
            training = tf.keras.backend.learning_phase()

        if not training or self.is_frozen():
            return self.__call__bn_frozen(inputs, training)
        else:
            return self.__call_with_bn(inputs, input_shape, training)

    def _apply_quantizer_if_defined(self, *, training, folded_weights):
        """
        Quantize weights if quantizer is defined
        :return: Quantized weights
        """
        if self.weights_quantizer is not None:
            folded_weights = self.weights_quantizer.__call__(folded_weights, training,
                                                             weights=self._quantizer_weights)
        return folded_weights
