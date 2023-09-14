# Project: Bachelor Thesis: Automated Quantization of Neural Networks
# Author: Miroslav Safar (xsafar23@stud.fit.vutbr.cz)

import tensorflow as tf
from keras.utils import control_flow_util

from tf_quantization.layers.base.quant_fused_conv2D_batch_norm_layer_base import \
    QuantFusedConv2DBatchNormalizationLayerBase


class ApproxQuantFusedConv2DBatchNormalizationLayer(QuantFusedConv2DBatchNormalizationLayerBase):
    """
    This class implements approximate method for solving problem with batch normalization folding during QAT
    for Conv2D + Batch Normalization
    """

    def __init__(self, filters, kernel_size, strides, padding, data_format, dilation_rate, groups, use_bias,
                 kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, kernel_constraint,
                 bias_constraint, axis, momentum, epsilon, center, scale, beta_initializer,
                 gamma_initializer, moving_mean_initializer, moving_variance_initializer, beta_regularizer,
                 gamma_regularizer, beta_constraint, gamma_constraint, quantize, quantize_num_bits_weight,
                 per_channel, symmetric, quantize_outputs=False, quantize_num_bits_output=8, **kwargs):
        super().__init__(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                         data_format=data_format, dilation_rate=dilation_rate, groups=groups, use_bias=use_bias,
                         kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint, axis=axis, momentum=momentum, epsilon=epsilon, center=center,
                         scale=scale, beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
                         moving_mean_initializer=moving_mean_initializer,
                         moving_variance_initializer=moving_variance_initializer, beta_regularizer=beta_regularizer,
                         gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint,
                         gamma_constraint=gamma_constraint, quantize=quantize,
                         quantize_num_bits_weight=quantize_num_bits_weight,
                         per_channel=per_channel, symmetric=symmetric, quantize_outputs=quantize_outputs,
                         quantize_num_bits_output=quantize_num_bits_output, **kwargs)

    def _reset_folded_weights(self, std_dev, outputs):
        """
        Resets folded weights

        outputs = ( std_dev / gamma) * outputs_folded

        :param std_dev: Standard deviation
        :param outputs: Outputs from conv with folded weights
        :return: Reseted outputs
        """
        gamma = tf.reshape(self.gamma, (1, 1, 1, self.gamma.shape[0]))
        std_dev = tf.reshape(std_dev, (1, 1, 1, std_dev.shape[0]))
        return (std_dev / gamma) * outputs

    def _call_bn_frozen(self, inputs, input_shape, training):
        """
        Execution graph for validation and training with frozen batch normalization
        """
        moving_std_dev = tf.math.sqrt(self.moving_variance + self.epsilon)
        if not self.per_channel:
            folded_weights = self._get_folded_weights(std_dev=moving_std_dev, kernel=self.kernel)
            folded_weights = self._apply_quantizer_if_defined(training=training, folded_weights=folded_weights)
        else:
            folded_weights = self.kernel
            folded_weights = self._apply_quantizer_if_defined(training=training, folded_weights=folded_weights)
            folded_weights = self._get_folded_weights(std_dev=moving_std_dev, kernel=folded_weights)

        outputs = self.convolution_op(inputs, folded_weights)
        if self.use_bias:
            outputs = self._add_folded_bias(outputs, self.bias, self.moving_mean, moving_std_dev)
        else:
            outputs = self._add_folded_bias(outputs, [0], self.moving_mean, moving_std_dev)

        if not tf.executing_eagerly() and input_shape.rank:
            # Infer the static output shape:
            out_shape = self.compute_output_shape(input_shape)
            outputs.set_shape(out_shape)

        return self._apply_outputs_quantizer_if_defined(outputs=outputs, training=training)

    def _call_with_bn(self, inputs, input_shape, training):
        """
        Execution graph for training with not fronzen batch normalozation
        """
        moving_std_dev = tf.math.sqrt(self.moving_variance + self.epsilon)

        if not self.per_channel:
            # We do folding and resetting only for per-tensor quantization
            # because it is not needed for per-channel quantization
            folded_weights = self._get_folded_weights(std_dev=moving_std_dev, kernel=self.kernel)
        else:
            folded_weights = self.kernel

        folded_weights = self._apply_quantizer_if_defined(training=training, folded_weights=folded_weights)

        outputs = self.convolution_op(inputs, folded_weights)

        if not self.per_channel:
            # * ( sqrt(var) / gamma)
            outputs = self._reset_folded_weights(moving_std_dev, outputs)

        if self.use_bias:
            outputs = tf.nn.bias_add(
                outputs, self.bias, data_format=self._tf_data_format
            )

        bn_input_shape = outputs.shape
        ndims = len(bn_input_shape)
        reduction_axes = [i for i in range(ndims) if i not in self.axis]
        batch_mean, batch_variance = tf.nn.moments(outputs, reduction_axes, keepdims=len(self.axis) > 1)

        # Update moving mean and variance
        new_mean, new_variance = batch_mean, batch_variance

        def _do_update(var, value):
            """
            Compute the updates for mean and variance.
            From: https://github.com/keras-team/keras
            """
            return self._assign_moving_average(
                var, value, self.momentum, input_shape[0]
            )

        def mean_update():
            """
            Update the moving mean
            From: https://github.com/keras-team/keras
            """
            true_branch = lambda: _do_update(self.moving_mean, new_mean)
            false_branch = lambda: self.moving_mean
            return control_flow_util.smart_cond(
                training, true_branch, false_branch
            )

        def variance_update():
            """
            Update the moving variance.
            From: https://github.com/keras-team/keras
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

        return self._apply_outputs_quantizer_if_defined(outputs=outputs, training=training)
