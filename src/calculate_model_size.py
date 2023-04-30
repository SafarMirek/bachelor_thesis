# Project: Bachelor Thesis: Automated Quantization of Neural Networks
# Author: Miroslav Safar (xsafar23@fit.vutbr.cz)

import keras.layers
import numpy as np
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapperV2

from tf_quantization.layers.approx.quant_conv2D_batch_layer import ApproxQuantFusedConv2DBatchNormalizationLayer
from tf_quantization.layers.approx.quant_depthwise_conv2d_bn_layer import \
    ApproxQuantFusedDepthwiseConv2DBatchNormalizationLayer
from tf_quantization.layers.quant_conv2D_batch_layer import QuantFusedConv2DBatchNormalizationLayer
from tf_quantization.layers.quant_depthwise_conv2d_bn_layer import QuantFusedDepthwiseConv2DBatchNormalizationLayer


def calculate_weights_mobilenet_size(model, per_channel=True, symmetric=True, only_layers=None):
    """
    This functions calculates memory size of weights of the original/quantized model
    Supported layers are: Dense Layer, Conv2D Layer, DepthwiseConv2DLayer, QuantConv2DBatch Layers
    and QuantDepthwiseConv2DBatchNormalization Layers
    :param model: Model to be analyzed
    :param per_channel: If quantization of weights is per_channel
    :param symmetric: If quantization of weights is symmetric
    :param only_layers: Calculate size of specific layers
    :return: model's weights memory size in bits
    """
    size = 0  # Model size in bits
    for layer in model.layers:
        if only_layers is not None and layer.name not in only_layers:
            continue
        layer_size = 0
        if isinstance(layer, QuantFusedConv2DBatchNormalizationLayer) or isinstance(layer, ApproxQuantFusedConv2DBatchNormalizationLayer):
            num_bits_weight = layer.quantize_num_bits_weight
            layer_size = layer_size + num_bits_weight * np.prod(layer.kernel.shape)

            # Quant koeficients
            if per_channel:
                layer_size = layer_size + 32 * (layer.kernel.shape[3])
                if not symmetric:
                    layer_size = layer_size + num_bits_weight * (layer.kernel.shape[3])
            else:
                layer_size = layer_size + 32
                if not symmetric:
                    layer_size = layer_size + num_bits_weight

            # add bias size (non quantized) it will be there every time because of batch norm fold
            layer_size = layer_size + 32 * (layer.kernel.shape[3])
        elif (
                isinstance(layer, QuantFusedDepthwiseConv2DBatchNormalizationLayer) or
                isinstance(layer, ApproxQuantFusedDepthwiseConv2DBatchNormalizationLayer)
        ):
            num_bits_weight = layer.quantize_num_bits_weight

            layer_size = layer_size + num_bits_weight * np.prod(layer.depthwise_kernel.shape)

            # Quant koeficients
            if per_channel:
                layer_size = layer_size + 32 * (layer.depthwise_kernel.shape[2])
                if not symmetric:
                    layer_size = layer_size + num_bits_weight * (layer.depthwise_kernel.shape[2])
            else:
                layer_size = layer_size + 32
                if not symmetric:
                    layer_size = layer_size + num_bits_weight

            # add bias size (non quantized) it will be there every time because of batch norm fold
            layer_size = layer_size + 32 * (layer.depthwise_kernel.shape[2])
        elif isinstance(layer, keras.layers.Dense) or isinstance(layer, keras.layers.Conv2D):
            layer_size = layer_size + 32 * np.prod(layer.kernel.shape)

            if layer.use_bias:
                layer_size = layer_size + 32 * np.prod(layer.bias.shape)
        elif isinstance(layer, keras.layers.DepthwiseConv2D):
            layer_size = layer_size + 32 * np.prod(layer.depthwise_kernel.shape)

            if layer.use_bias:
                layer_size = layer_size + 32 * np.prod(layer.bias.shape)
        elif isinstance(layer, QuantizeWrapperV2):
            num_bits_weight = 8
            if "num_bits_weight" in layer.quantize_config.get_config():
                num_bits_weight = layer.quantize_config.get_config()["num_bits_weight"]
            if isinstance(layer.layer, keras.layers.Conv2D):
                # kernel
                # (bits_per_weight) * (number_of_weights) +
                # + (quantization constant 32 bit) * (if per_channel (channels) else 1)
                layer_size = layer_size + num_bits_weight * np.prod(layer.layer.kernel.shape)

                if per_channel:
                    layer_size = layer_size + 32 * (layer.layer.kernel.shape[len(layer.layer.kernel.shape) - 1])
                    if not symmetric:
                        layer_size = layer_size + num_bits_weight * (
                            layer.layer.kernel.shape[len(layer.layer.kernel.shape) - 1])
                else:
                    layer_size = layer_size + 32
                    if not symmetric:
                        layer_size = layer_size + num_bits_weight

                if layer.layer.use_bias:
                    layer_size = layer_size + 32 * np.prod(layer.layer.bias.shape)  # Bias is not quantized
            elif isinstance(layer.layer, keras.layers.Dense):
                # kernel
                layer_size = layer_size + num_bits_weight * np.prod(layer.layer.kernel.shape)

                layer_size = layer_size + 32  # Scaling factor
                if not symmetric:
                    layer_size = layer_size + num_bits_weight  # Zero-point

                if layer.layer.use_bias:
                    layer_size = layer_size + 32 * np.prod(layer.layer.bias.shape)  # Bias is not quantized
        size = size + layer_size
    return size
