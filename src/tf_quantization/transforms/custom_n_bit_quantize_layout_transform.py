# Project: Bachelor Thesis: Automated Quantization of Neural Networks
# Author: Miroslav Safar (xsafar23@stud.fit.vutbr.cz)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras import quantize_layout_transform
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit import \
    default_n_bit_transforms
from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import model_transformer
from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import transforms

from tf_quantization.layers.approx.quant_conv2D_batch_layer import ApproxQuantFusedConv2DBatchNormalizationLayer
from tf_quantization.layers.approx.quant_depthwise_conv2d_bn_layer import \
    ApproxQuantFusedDepthwiseConv2DBatchNormalizationLayer
from tf_quantization.layers.quant_conv2D_batch_layer import QuantFusedConv2DBatchNormalizationLayer
from tf_quantization.layers.quant_depthwise_conv2d_bn_layer import QuantFusedDepthwiseConv2DBatchNormalizationLayer

keras = tf.keras
LayerNode = transforms.LayerNode
LayerPattern = transforms.LayerPattern


class CustomNBitQuantizeLayoutTransform(quantize_layout_transform.QuantizeLayoutTransform):
    """
    This transform is equivalent to DefaultNBitQuantizeLayoutTransform but adds
    DepthwiseConv2DBatchNormReluQuantize and Conv2DBatchNormReluQuantize transforms
    """

    def __init__(self, only_layers, num_bits_weight: int = 8, num_bits_activation: int = 8, approx=False,
                 symmetric=True, per_channel=True):
        self.only_layers = only_layers
        self._num_bits_weight = num_bits_weight
        self._num_bits_activation = num_bits_activation
        self._approx = approx
        self._symmetric = symmetric
        self._per_channel = per_channel

        if self._per_channel and not self._approx:
            raise ValueError("It does not make sense to use not approx scheme with per_channel quantization")

    def apply(self, model, layer_quantize_map):
        """
        Implement default 8-bit transforms and
        our DepthwiseConv2DBatchNormReluQuantize and Conv2DBatchNormReluQuantize transforms

        Currently this function handles Fusing Conv2D/DepthwiseConv2D (+ BN) + ReLU into single layer
        by ensuring quantization is not applied between them

        Args:
          model: Keras model to be quantized.
          layer_quantize_map: Map with keys as layer names, and values as dicts
            containing custom `QuantizeConfig`s which may have been passed with
            layers.

        Returns:
          Transformed model where quantization better match deployment
        """

        transforms = [
            DepthwiseConv2DBatchNormReluQuantize(
                num_bits_weight=self._num_bits_weight,
                num_bits_activation=self._num_bits_activation,
                approx=self._approx,
                symmetric=self._symmetric,
                per_channel=self._per_channel
            ),
            Conv2DBatchNormReluQuantize(
                num_bits_weight=self._num_bits_weight,
                num_bits_activation=self._num_bits_activation,
                approx=self._approx,
                symmetric=self._symmetric,
                per_channel=self._per_channel
            ),
            Conv2DBatchNormQuantize(
                num_bits_weight=self._num_bits_weight,
                num_bits_activation=self._num_bits_activation,
                approx=self._approx,
                symmetric=self._symmetric,
                per_channel=self._per_channel
            ),
            default_n_bit_transforms.InputLayerQuantize(
                num_bits_weight=self._num_bits_weight,
                num_bits_activation=self._num_bits_activation),
            default_n_bit_transforms.SeparableConv1DQuantize(
                num_bits_weight=self._num_bits_weight,
                num_bits_activation=self._num_bits_activation),
            default_n_bit_transforms.SeparableConvQuantize(
                num_bits_weight=self._num_bits_weight,
                num_bits_activation=self._num_bits_activation),
            default_n_bit_transforms.Conv2DReshapeBatchNormReLUQuantize(
                num_bits_weight=self._num_bits_weight,
                num_bits_activation=self._num_bits_activation),
            default_n_bit_transforms.Conv2DReshapeBatchNormActivationQuantize(
                num_bits_weight=self._num_bits_weight,
                num_bits_activation=self._num_bits_activation),
            default_n_bit_transforms.Conv2DBatchNormReLUQuantize(
                num_bits_weight=self._num_bits_weight,
                num_bits_activation=self._num_bits_activation),
            default_n_bit_transforms.Conv2DBatchNormActivationQuantize(
                num_bits_weight=self._num_bits_weight,
                num_bits_activation=self._num_bits_activation),
            default_n_bit_transforms.Conv2DReshapeBatchNormQuantize(
                num_bits_weight=self._num_bits_weight,
                num_bits_activation=self._num_bits_activation),
            default_n_bit_transforms.Conv2DBatchNormQuantize(
                num_bits_weight=self._num_bits_weight,
                num_bits_activation=self._num_bits_activation),
            default_n_bit_transforms.ConcatTransform6Inputs(
                num_bits_weight=self._num_bits_weight,
                num_bits_activation=self._num_bits_activation),
            default_n_bit_transforms.ConcatTransform5Inputs(
                num_bits_weight=self._num_bits_weight,
                num_bits_activation=self._num_bits_activation),
            default_n_bit_transforms.ConcatTransform4Inputs(
                num_bits_weight=self._num_bits_weight,
                num_bits_activation=self._num_bits_activation),
            default_n_bit_transforms.ConcatTransform3Inputs(
                num_bits_weight=self._num_bits_weight,
                num_bits_activation=self._num_bits_activation),
            default_n_bit_transforms.ConcatTransform(
                num_bits_weight=self._num_bits_weight,
                num_bits_activation=self._num_bits_activation),
            default_n_bit_transforms.LayerReLUQuantize(
                num_bits_weight=self._num_bits_weight,
                num_bits_activation=self._num_bits_activation),
            default_n_bit_transforms.LayerReluActivationQuantize(
                num_bits_weight=self._num_bits_weight,
                num_bits_activation=self._num_bits_activation),
            default_n_bit_transforms.DenseBatchNormQuantize(
                num_bits_weight=self._num_bits_weight,
                num_bits_activation=self._num_bits_activation),
            default_n_bit_transforms.DenseBatchNormReLUQuantize(
                num_bits_weight=self._num_bits_weight,
                num_bits_activation=self._num_bits_activation),
            default_n_bit_transforms.DenseBatchNormActivationQuantize(
                num_bits_weight=self._num_bits_weight,
                num_bits_activation=self._num_bits_activation),
        ]
        return model_transformer.ModelTransformer(
            model, transforms,
            set(self.only_layers), layer_quantize_map).transform()


class Conv2DBatchNormReluQuantize(transforms.Transform):
    """
    Transformation that replaces Conv2D + BN + ReLU with (Approx)QuantFusedConv2DBatchNormalizationLayer + ReLU

    This ensures batch normalization is handled properly if per-tensor weight quantization is used
    """

    def __init__(self, num_bits_weight: int = 8, num_bits_activation: int = 8, approx: bool = False,
                 symmetric: bool = True, per_channel: bool = True):
        self.num_bits_weight = num_bits_weight
        self.num_bits_activation = num_bits_activation
        self.approx = approx
        self.symmetric = symmetric
        self.per_channel = per_channel

    def pattern(self):
        return LayerPattern(
            'ReLU',
            inputs=[LayerPattern(
                'BatchNormalization',
                inputs=[LayerPattern('Conv2D')])]
        )

    def _replace(self, relu_layer_node, bn_layer_node, conv_layer_node):
        if _has_custom_quantize_config(
                relu_layer_node, bn_layer_node, conv_layer_node):
            return relu_layer_node

        if self.approx:
            conv_bn_layer = ApproxQuantFusedConv2DBatchNormalizationLayer(
                name=conv_layer_node.layer['config']['name'] + "_bnfolded",
                filters=conv_layer_node.layer['config']['filters'],
                kernel_size=conv_layer_node.layer['config']['kernel_size'],
                strides=conv_layer_node.layer['config']['strides'],
                padding=conv_layer_node.layer['config']['padding'],
                data_format=conv_layer_node.layer['config']['data_format'],
                dilation_rate=conv_layer_node.layer['config']['dilation_rate'],
                groups=conv_layer_node.layer['config']['groups'],
                use_bias=conv_layer_node.layer['config']['use_bias'],
                kernel_initializer=conv_layer_node.layer['config']['kernel_initializer'],
                bias_initializer=conv_layer_node.layer['config']['bias_initializer'],
                kernel_regularizer=conv_layer_node.layer['config']['kernel_regularizer'],
                bias_regularizer=conv_layer_node.layer['config']['bias_regularizer'],
                kernel_constraint=conv_layer_node.layer['config']['kernel_constraint'],
                bias_constraint=conv_layer_node.layer['config']['bias_constraint'],
                axis=bn_layer_node.layer['config']['axis'],
                momentum=bn_layer_node.layer['config']['momentum'],
                epsilon=bn_layer_node.layer['config']['epsilon'],
                center=bn_layer_node.layer['config']['center'],
                scale=bn_layer_node.layer['config']['scale'],
                beta_initializer=bn_layer_node.layer['config']['beta_initializer'],
                gamma_initializer=bn_layer_node.layer['config']['gamma_initializer'],
                moving_mean_initializer=bn_layer_node.layer['config']['moving_mean_initializer'],
                moving_variance_initializer=bn_layer_node.layer['config']['moving_variance_initializer'],
                beta_regularizer=bn_layer_node.layer['config']['beta_regularizer'],
                gamma_regularizer=bn_layer_node.layer['config']['gamma_regularizer'],
                beta_constraint=bn_layer_node.layer['config']['beta_constraint'],
                gamma_constraint=bn_layer_node.layer['config']['gamma_constraint'],
                quantize=True,
                quantize_num_bits_weight=self.num_bits_weight,
                symmetric=self.symmetric,
                per_channel=self.per_channel
            )
        else:
            conv_bn_layer = QuantFusedConv2DBatchNormalizationLayer(
                name=conv_layer_node.layer['config']['name'] + "_bnfolded",
                filters=conv_layer_node.layer['config']['filters'],
                kernel_size=conv_layer_node.layer['config']['kernel_size'],
                strides=conv_layer_node.layer['config']['strides'],
                padding=conv_layer_node.layer['config']['padding'],
                data_format=conv_layer_node.layer['config']['data_format'],
                dilation_rate=conv_layer_node.layer['config']['dilation_rate'],
                groups=conv_layer_node.layer['config']['groups'],
                use_bias=conv_layer_node.layer['config']['use_bias'],
                kernel_initializer=conv_layer_node.layer['config']['kernel_initializer'],
                bias_initializer=conv_layer_node.layer['config']['bias_initializer'],
                kernel_regularizer=conv_layer_node.layer['config']['kernel_regularizer'],
                bias_regularizer=conv_layer_node.layer['config']['bias_regularizer'],
                kernel_constraint=conv_layer_node.layer['config']['kernel_constraint'],
                bias_constraint=conv_layer_node.layer['config']['bias_constraint'],
                axis=bn_layer_node.layer['config']['axis'],
                momentum=bn_layer_node.layer['config']['momentum'],
                epsilon=bn_layer_node.layer['config']['epsilon'],
                center=bn_layer_node.layer['config']['center'],
                scale=bn_layer_node.layer['config']['scale'],
                beta_initializer=bn_layer_node.layer['config']['beta_initializer'],
                gamma_initializer=bn_layer_node.layer['config']['gamma_initializer'],
                moving_mean_initializer=bn_layer_node.layer['config']['moving_mean_initializer'],
                moving_variance_initializer=bn_layer_node.layer['config']['moving_variance_initializer'],
                beta_regularizer=bn_layer_node.layer['config']['beta_regularizer'],
                gamma_regularizer=bn_layer_node.layer['config']['gamma_regularizer'],
                beta_constraint=bn_layer_node.layer['config']['beta_constraint'],
                gamma_constraint=bn_layer_node.layer['config']['gamma_constraint'],
                quantize=True,
                quantize_num_bits_weight=self.num_bits_weight,
                symmetric=self.symmetric,
                per_channel=self.per_channel
            )

        conv_bn_layer_config = keras.layers.serialize(conv_bn_layer)
        conv_bn_layer_config['name'] = conv_bn_layer.name

        conv_layer_weights = conv_layer_node.weights
        bn_layer_weights = bn_layer_node.weights
        conv_bn_layer_weights = collections.OrderedDict()
        conv_bn_layer_weights['kernel:0'] = conv_layer_weights['kernel:0']
        if conv_bn_layer.use_bias:
            conv_bn_layer_weights['bias:0'] = conv_layer_weights['bias:0']

        if conv_bn_layer.scale:
            conv_bn_layer_weights['gamma:0'] = bn_layer_weights['gamma:0']

        if conv_bn_layer.center:
            conv_bn_layer_weights['beta:0'] = bn_layer_weights['beta:0']

        conv_bn_layer_weights['moving_mean:0'] = bn_layer_weights['moving_mean:0']
        conv_bn_layer_weights['moving_variance:0'] = bn_layer_weights['moving_variance:0']

        relu_layer_node.input_layers = [
            LayerNode(
                conv_bn_layer_config,
                weights=conv_bn_layer_weights,
                input_layers=conv_layer_node.input_layers,
                metadata={}
            )
        ]

        return relu_layer_node

    def replacement(self, match_layer):
        relu_layer_node = match_layer
        bn_layer_node = relu_layer_node.input_layers[0]
        conv_layer_node = bn_layer_node.input_layers[0]

        return self._replace(relu_layer_node, bn_layer_node, conv_layer_node)


class Conv2DBatchNormQuantize(transforms.Transform):
    """
    Transformation that replaces Conv2D + BN + ReLU with (Approx)QuantFusedConv2DBatchNormalizationLayer + ReLU

    This ensures batch normalization is handled properly if per-tensor weight quantization is used
    """

    def __init__(self, num_bits_weight: int = 8, num_bits_activation: int = 8, approx: bool = False,
                 symmetric: bool = True, per_channel: bool = True):
        self.num_bits_weight = num_bits_weight
        self.num_bits_activation = num_bits_activation
        self.approx = approx
        self.symmetric = symmetric
        self.per_channel = per_channel

    def pattern(self):
        return LayerPattern(
            'BatchNormalization',
            inputs=[LayerPattern('Conv2D')])

    def _replace(self, bn_layer_node, conv_layer_node):
        if _has_custom_quantize_config(bn_layer_node, conv_layer_node):
            return bn_layer_node

        if self.approx:
            conv_bn_layer = ApproxQuantFusedConv2DBatchNormalizationLayer(
                name=conv_layer_node.layer['config']['name'] + "_bnfolded",
                filters=conv_layer_node.layer['config']['filters'],
                kernel_size=conv_layer_node.layer['config']['kernel_size'],
                strides=conv_layer_node.layer['config']['strides'],
                padding=conv_layer_node.layer['config']['padding'],
                data_format=conv_layer_node.layer['config']['data_format'],
                dilation_rate=conv_layer_node.layer['config']['dilation_rate'],
                groups=conv_layer_node.layer['config']['groups'],
                use_bias=conv_layer_node.layer['config']['use_bias'],
                kernel_initializer=conv_layer_node.layer['config']['kernel_initializer'],
                bias_initializer=conv_layer_node.layer['config']['bias_initializer'],
                kernel_regularizer=conv_layer_node.layer['config']['kernel_regularizer'],
                bias_regularizer=conv_layer_node.layer['config']['bias_regularizer'],
                kernel_constraint=conv_layer_node.layer['config']['kernel_constraint'],
                bias_constraint=conv_layer_node.layer['config']['bias_constraint'],
                axis=bn_layer_node.layer['config']['axis'],
                momentum=bn_layer_node.layer['config']['momentum'],
                epsilon=bn_layer_node.layer['config']['epsilon'],
                center=bn_layer_node.layer['config']['center'],
                scale=bn_layer_node.layer['config']['scale'],
                beta_initializer=bn_layer_node.layer['config']['beta_initializer'],
                gamma_initializer=bn_layer_node.layer['config']['gamma_initializer'],
                moving_mean_initializer=bn_layer_node.layer['config']['moving_mean_initializer'],
                moving_variance_initializer=bn_layer_node.layer['config']['moving_variance_initializer'],
                beta_regularizer=bn_layer_node.layer['config']['beta_regularizer'],
                gamma_regularizer=bn_layer_node.layer['config']['gamma_regularizer'],
                beta_constraint=bn_layer_node.layer['config']['beta_constraint'],
                gamma_constraint=bn_layer_node.layer['config']['gamma_constraint'],
                quantize=True,
                quantize_num_bits_weight=self.num_bits_weight,
                symmetric=self.symmetric,
                per_channel=self.per_channel,
                quantize_outputs=True,
                quantize_num_bits_output=self.num_bits_activation

            )
        else:
            conv_bn_layer = QuantFusedConv2DBatchNormalizationLayer(
                name=conv_layer_node.layer['config']['name'] + "_bnfolded",
                filters=conv_layer_node.layer['config']['filters'],
                kernel_size=conv_layer_node.layer['config']['kernel_size'],
                strides=conv_layer_node.layer['config']['strides'],
                padding=conv_layer_node.layer['config']['padding'],
                data_format=conv_layer_node.layer['config']['data_format'],
                dilation_rate=conv_layer_node.layer['config']['dilation_rate'],
                groups=conv_layer_node.layer['config']['groups'],
                use_bias=conv_layer_node.layer['config']['use_bias'],
                kernel_initializer=conv_layer_node.layer['config']['kernel_initializer'],
                bias_initializer=conv_layer_node.layer['config']['bias_initializer'],
                kernel_regularizer=conv_layer_node.layer['config']['kernel_regularizer'],
                bias_regularizer=conv_layer_node.layer['config']['bias_regularizer'],
                kernel_constraint=conv_layer_node.layer['config']['kernel_constraint'],
                bias_constraint=conv_layer_node.layer['config']['bias_constraint'],
                axis=bn_layer_node.layer['config']['axis'],
                momentum=bn_layer_node.layer['config']['momentum'],
                epsilon=bn_layer_node.layer['config']['epsilon'],
                center=bn_layer_node.layer['config']['center'],
                scale=bn_layer_node.layer['config']['scale'],
                beta_initializer=bn_layer_node.layer['config']['beta_initializer'],
                gamma_initializer=bn_layer_node.layer['config']['gamma_initializer'],
                moving_mean_initializer=bn_layer_node.layer['config']['moving_mean_initializer'],
                moving_variance_initializer=bn_layer_node.layer['config']['moving_variance_initializer'],
                beta_regularizer=bn_layer_node.layer['config']['beta_regularizer'],
                gamma_regularizer=bn_layer_node.layer['config']['gamma_regularizer'],
                beta_constraint=bn_layer_node.layer['config']['beta_constraint'],
                gamma_constraint=bn_layer_node.layer['config']['gamma_constraint'],
                quantize=True,
                quantize_num_bits_weight=self.num_bits_weight,
                symmetric=self.symmetric,
                per_channel=self.per_channel,
                quantize_outputs=True,
                quantize_num_bits_output=self.num_bits_activation
            )

        conv_bn_layer_config = keras.layers.serialize(conv_bn_layer)
        conv_bn_layer_config['name'] = conv_bn_layer.name

        conv_layer_weights = conv_layer_node.weights
        bn_layer_weights = bn_layer_node.weights
        conv_bn_layer_weights = collections.OrderedDict()
        conv_bn_layer_weights['kernel:0'] = conv_layer_weights['kernel:0']
        if conv_bn_layer.use_bias:
            conv_bn_layer_weights['bias:0'] = conv_layer_weights['bias:0']

        if conv_bn_layer.scale:
            conv_bn_layer_weights['gamma:0'] = bn_layer_weights['gamma:0']

        if conv_bn_layer.center:
            conv_bn_layer_weights['beta:0'] = bn_layer_weights['beta:0']

        conv_bn_layer_weights['moving_mean:0'] = bn_layer_weights['moving_mean:0']
        conv_bn_layer_weights['moving_variance:0'] = bn_layer_weights['moving_variance:0']

        return LayerNode(
            conv_bn_layer_config,
            weights=conv_bn_layer_weights,
            input_layers=conv_layer_node.input_layers,
            metadata={}
        )

    def replacement(self, match_layer):
        bn_layer_node = match_layer
        conv_layer_node = bn_layer_node.input_layers[0]

        return self._replace(bn_layer_node, conv_layer_node)


class DepthwiseConv2DBatchNormReluQuantize(transforms.Transform):
    """
    Transformation that replaces DepthwiseConv2D + BN + ReLU with
    (Approx)QuantFusedDepthwiseConv2DBatchNormalizationLayer + ReLU

    This ensures batch normalization is handled properly if per-tensor weight quantization is used
    """

    def __init__(self, num_bits_weight: int = 8, num_bits_activation: int = 8, approx: bool = False,
                 symmetric: bool = True, per_channel: bool = True):
        self.num_bits_weight = num_bits_weight
        self.num_bits_activation = num_bits_activation
        self.approx = approx
        self.symmetric = symmetric
        self.per_channel = per_channel

    def pattern(self):
        return LayerPattern(
            'ReLU',
            inputs=[LayerPattern(
                'BatchNormalization',
                inputs=[LayerPattern('DepthwiseConv2D')])]
        )

    def _replace(self, relu_layer_node, bn_layer_node, conv_layer_node):
        if _has_custom_quantize_config(
                relu_layer_node, bn_layer_node, conv_layer_node):
            return relu_layer_node

        if self.approx:
            conv_bn_layer = ApproxQuantFusedDepthwiseConv2DBatchNormalizationLayer(
                name=conv_layer_node.layer['config']['name'] + "_bnfolded",
                kernel_size=conv_layer_node.layer['config']['kernel_size'],
                strides=conv_layer_node.layer['config']['strides'],
                padding=conv_layer_node.layer['config']['padding'],
                depth_multiplier=conv_layer_node.layer['config']['depth_multiplier'],
                data_format=conv_layer_node.layer['config']['data_format'],
                dilation_rate=conv_layer_node.layer['config']['dilation_rate'],
                activation=conv_layer_node.layer['config']['activation'],
                use_bias=conv_layer_node.layer['config']['use_bias'],
                depthwise_initializer=conv_layer_node.layer['config']['depthwise_initializer'],
                bias_initializer=conv_layer_node.layer['config']['bias_initializer'],
                depthwise_regularizer=conv_layer_node.layer['config']['depthwise_regularizer'],
                bias_regularizer=conv_layer_node.layer['config']['bias_regularizer'],
                activity_regularizer=conv_layer_node.layer['config']['activity_regularizer'],
                depthwise_constraint=conv_layer_node.layer['config']['depthwise_constraint'],
                bias_constraint=conv_layer_node.layer['config']['bias_constraint'],
                axis=bn_layer_node.layer['config']['axis'],
                momentum=bn_layer_node.layer['config']['momentum'],
                epsilon=bn_layer_node.layer['config']['epsilon'],
                center=bn_layer_node.layer['config']['center'],
                scale=bn_layer_node.layer['config']['scale'],
                beta_initializer=bn_layer_node.layer['config']['beta_initializer'],
                gamma_initializer=bn_layer_node.layer['config']['gamma_initializer'],
                moving_mean_initializer=bn_layer_node.layer['config']['moving_mean_initializer'],
                moving_variance_initializer=bn_layer_node.layer['config']['moving_variance_initializer'],
                beta_regularizer=bn_layer_node.layer['config']['beta_regularizer'],
                gamma_regularizer=bn_layer_node.layer['config']['gamma_regularizer'],
                beta_constraint=bn_layer_node.layer['config']['beta_constraint'],
                gamma_constraint=bn_layer_node.layer['config']['gamma_constraint'],
                quantize=True,
                quantize_num_bits_weight=self.num_bits_weight,
                symmetric=self.symmetric,
                per_channel=self.per_channel
            )
        else:
            conv_bn_layer = QuantFusedDepthwiseConv2DBatchNormalizationLayer(
                name=conv_layer_node.layer['config']['name'] + "_bnfolded",
                kernel_size=conv_layer_node.layer['config']['kernel_size'],
                strides=conv_layer_node.layer['config']['strides'],
                padding=conv_layer_node.layer['config']['padding'],
                depth_multiplier=conv_layer_node.layer['config']['depth_multiplier'],
                data_format=conv_layer_node.layer['config']['data_format'],
                dilation_rate=conv_layer_node.layer['config']['dilation_rate'],
                activation=conv_layer_node.layer['config']['activation'],
                use_bias=conv_layer_node.layer['config']['use_bias'],
                depthwise_initializer=conv_layer_node.layer['config']['depthwise_initializer'],
                bias_initializer=conv_layer_node.layer['config']['bias_initializer'],
                depthwise_regularizer=conv_layer_node.layer['config']['depthwise_regularizer'],
                bias_regularizer=conv_layer_node.layer['config']['bias_regularizer'],
                activity_regularizer=conv_layer_node.layer['config']['activity_regularizer'],
                depthwise_constraint=conv_layer_node.layer['config']['depthwise_constraint'],
                bias_constraint=conv_layer_node.layer['config']['bias_constraint'],
                axis=bn_layer_node.layer['config']['axis'],
                momentum=bn_layer_node.layer['config']['momentum'],
                epsilon=bn_layer_node.layer['config']['epsilon'],
                center=bn_layer_node.layer['config']['center'],
                scale=bn_layer_node.layer['config']['scale'],
                beta_initializer=bn_layer_node.layer['config']['beta_initializer'],
                gamma_initializer=bn_layer_node.layer['config']['gamma_initializer'],
                moving_mean_initializer=bn_layer_node.layer['config']['moving_mean_initializer'],
                moving_variance_initializer=bn_layer_node.layer['config']['moving_variance_initializer'],
                beta_regularizer=bn_layer_node.layer['config']['beta_regularizer'],
                gamma_regularizer=bn_layer_node.layer['config']['gamma_regularizer'],
                beta_constraint=bn_layer_node.layer['config']['beta_constraint'],
                gamma_constraint=bn_layer_node.layer['config']['gamma_constraint'],
                quantize=True,
                quantize_num_bits_weight=self.num_bits_weight,
                symmetric=self.symmetric,
                per_channel=self.per_channel
            )

        conv_bn_layer_config = keras.layers.serialize(conv_bn_layer)
        conv_bn_layer_config['name'] = conv_bn_layer.name

        conv_layer_weights = conv_layer_node.weights
        bn_layer_weights = bn_layer_node.weights
        conv_bn_layer_weights = collections.OrderedDict()
        conv_bn_layer_weights['depthwise_kernel:0'] = conv_layer_weights['depthwise_kernel:0']
        if conv_bn_layer.use_bias:
            conv_bn_layer_weights['bias:0'] = conv_layer_weights['bias:0']

        if conv_bn_layer.scale:
            conv_bn_layer_weights['gamma:0'] = bn_layer_weights['gamma:0']

        if conv_bn_layer.center:
            conv_bn_layer_weights['beta:0'] = bn_layer_weights['beta:0']

        conv_bn_layer_weights['moving_mean:0'] = bn_layer_weights['moving_mean:0']
        conv_bn_layer_weights['moving_variance:0'] = bn_layer_weights['moving_variance:0']

        relu_layer_node.input_layers = [
            LayerNode(
                conv_bn_layer_config,
                weights=conv_bn_layer_weights,
                input_layers=conv_layer_node.input_layers,
                metadata={}
            )
        ]

        return relu_layer_node

    def replacement(self, match_layer):
        relu_layer_node = match_layer
        bn_layer_node = relu_layer_node.input_layers[0]
        conv_layer_node = bn_layer_node.input_layers[0]

        return self._replace(relu_layer_node, bn_layer_node, conv_layer_node)


def _get_quantize_config(layer_node):
    """Extracts quantization config from layer metadata"""
    return layer_node.metadata.get('quantize_config')


def _has_custom_quantize_config(*layer_nodes):
    """Checks if layer has quantization config in its metadata"""
    for layer_node in layer_nodes:
        if _get_quantize_config(layer_node) is not None:
            return True
    return False
