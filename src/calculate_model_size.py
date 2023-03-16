import keras.layers
import numpy as np
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapperV2

from tf_quantization.layers.quant_conv2D_batch_layer import QuantConv2DBatchLayer
from tf_quantization.layers.quant_depthwise_conv2d_bn_layer import QuantDepthwiseConv2DBatchNormalizationLayer


def calculate_weights_mobilenet_size(model, per_channel=True):
    size = 0  # Model size in bits
    for layer in model.layers:
        layer_size = 0
        if isinstance(layer, keras.layers.Dense) or isinstance(layer, keras.layers.Conv2D):
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
                else:
                    layer_size = layer_size + 32

                if layer.layer.use_bias:
                    layer_size = layer_size + 32 * np.prod(layer.layer.bias.shape)  # Bias is not quantized
            elif isinstance(layer.layer, keras.layers.Dense):
                # kernel
                layer_size = layer_size + num_bits_weight * np.prod(layer.layer.kernel.shape)

                layer_size = layer_size + 32

                if layer.layer.use_bias:
                    layer_size = layer_size + 32 * np.prod(layer.layer.bias.shape)  # Bias is not quantized
        elif isinstance(layer, QuantConv2DBatchLayer):
            num_bits_weight = layer.quantize_num_bits_weight
            layer_size = layer_size + num_bits_weight * np.prod(layer.kernel.shape)

            # Quant koeficients
            layer_size = layer_size + 32 * (layer.kernel.shape[3])

            # add bias size (non quantized) it will be there every time because of batch norm fold
            layer_size = layer_size + 32 * (layer.kernel.shape[3])
        elif isinstance(layer, QuantDepthwiseConv2DBatchNormalizationLayer):
            num_bits_weight = layer.quantize_num_bits_weight

            layer_size = layer_size + num_bits_weight * np.prod(layer.depthwise_kernel.shape)

            # Quant koeficients
            layer_size = layer_size + 32 * (layer.depthwise_kernel.shape[2])

            # add bias size (non quantized) it will be there every time because of batch norm fold
            layer_size = layer_size + 32 * (layer.depthwise_kernel.shape[2])
        print(f"Layer {layer.name}: {layer_size}")
        size = size + layer_size
    return size
