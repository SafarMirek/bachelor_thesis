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


class QuantDepthwiseConv2DBatchNormalizationLayer(keras.layers.DepthwiseConv2D, keras.layers.BatchNormalization):

    def __init__(self, dephwise_conv_layer: keras.layers.DepthwiseConv2D, bn_layer: keras.layers.BatchNormalization,
                 quantize=True, **kwargs):
        keras.layers.DepthwiseConv2D.__init__(self, **dephwise_conv_layer.get_config())
        keras.layers.BatchNormalization.__init__(self, **bn_layer.get_config())
        self._dephwise_conv_layer = dephwise_conv_layer
        self._bn_layer = bn_layer

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
        keras.layers.BatchNormalization.build(self, input_shape)

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
            outputs, bias, data_format=self.data_format
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
                conv_out, self.bias, data_format=self.data_format
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
