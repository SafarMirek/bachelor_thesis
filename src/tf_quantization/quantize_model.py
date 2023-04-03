from tensorflow_model_optimization.python.core.quantization.keras import quantize_scheme
from tensorflow_model_optimization.python.core.quantization.keras.quantize import quantize_annotate_model, \
    quantize_apply, quantize_scope

from tf_quantization.layers.approx.quant_conv2D_batch_layer import ApproximateQuantConv2DBatchLayer
from tf_quantization.layers.approx.quant_depthwise_conv2d_bn_layer import \
    ApproximateQuantDepthwiseConv2DBatchNormalizationLayer
from tf_quantization.layers.quant_conv2D_batch_layer import QuantConv2DBatchLayer
from tf_quantization.layers.quant_depthwise_conv2d_bn_layer import QuantDepthwiseConv2DBatchNormalizationLayer
from tf_quantization.quantize_registry import QuantizeRegistry
from tf_quantization.transforms.quantize_transforms import PerLayerQuantizeModelTransformer, \
    PerLayerQuantizeLayoutTransform


class PerLayerNBitQuantizeScheme(quantize_scheme.QuantizeScheme):
    """Default N-Bit Scheme supported by TFLite."""

    def __init__(self, transformer_fn, registry_fn):
        self.transformer_fn = transformer_fn
        self.registry_fn = registry_fn

    def get_layout_transformer(self):
        return self.transformer_fn()

    def get_quantize_registry(self):
        return self.registry_fn()


def quantize_model(model, quantization_config, activation_quant_no_affect=False, approx=False, per_channel=True,
                   symmetric=True):
    # TODO: Implement support for transforms, that changes layers (SeparableConv, etc...)

    with quantize_scope({
        "ApproximateQuantConv2DBatchLayer": ApproximateQuantConv2DBatchLayer,
        "QuantConv2DBatchLayer": QuantConv2DBatchLayer,
        "QuantDepthwiseConv2DBatchNormalizationLayer": QuantDepthwiseConv2DBatchNormalizationLayer,
        "ApproximateQuantDepthwiseConv2DBatchNormalizationLayer": ApproximateQuantDepthwiseConv2DBatchNormalizationLayer,

    }):
        transformer = PerLayerQuantizeModelTransformer(model, quantization_config, {}, approx=approx,
                                                       symmetric=symmetric, per_channel=per_channel)

        if transformer.get_number_of_quantizable_layers() != len(quantization_config):
            raise ValueError(
                f'There is different number of quantizable layers in model than in quantization config. ' +
                f'({transformer.get_number_of_quantizable_layers()} needed)')

        layer_group_map = transformer.get_layers_quantize_group_map()

        layer_quantization_config_map = {}
        for layer in layer_group_map.keys():
            layer_quantization_config_map[layer] = quantization_config[layer_group_map[layer]]

        scheme = PerLayerNBitQuantizeScheme(
            transformer_fn=lambda: PerLayerQuantizeLayoutTransform(quantization_config, approx=approx,
                                                                   symmetric=symmetric, per_channel=per_channel),
            registry_fn=lambda: QuantizeRegistry(
                layer_quantization_config_map,
                activation_quant_no_affect=activation_quant_no_affect,
                symmetric=symmetric,
                per_channel=per_channel
            )
        )

        annotated_model = quantize_annotate_model(model)
        return quantize_apply(annotated_model, scheme=scheme)
