from tensorflow_model_optimization.python.core.quantization.keras import quantize_scheme
from tensorflow_model_optimization.python.core.quantization.keras.quantize import quantize_annotate_model, \
    quantize_apply

from quantization_search.tf_quantization.quantize_registry import QuantizeRegistry
from quantization_search.tf_quantization.transforms.quantize_transforms import PerLayerQuantizeModelTransformer, \
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


def quantize_model(model, quantization_config):
    # TODO:
    # Provede transformaci modelu a pak udela copy modelu s tim, ze quantizuje vrstvy, ktere nekvantizoval transorfmer
    # pomocí configů z QuantizeRegistry
    # To by mělo být všechny pro vytvoření QA modelu

    # TODO: Implement support for transforms, that changes layers (SeparableConv, etc...)
    # TODO: Make graph transform to make better Conv+BN+Relu handling (implement some typ)

    transformer = PerLayerQuantizeModelTransformer(model, quantization_config)

    if transformer.get_number_of_quantizable_layers() != len(quantization_config):
        raise ValueError(f'There is different number of quantizable layers in model than in quantization config. ({transformer.get_number_of_quantizable_layers()} needed)')

    layer_group_map = transformer.get_layers_quantize_group_map()
    layer_quantization_config_map = {}
    for layer in layer_group_map.keys():
        layer_quantization_config_map[layer] = quantization_config[layer_group_map[layer]]

    scheme = PerLayerNBitQuantizeScheme(transformer_fn=lambda: PerLayerQuantizeLayoutTransform(quantization_config),
                                        registry_fn=lambda: QuantizeRegistry(layer_quantization_config_map))

    annotated_model = quantize_annotate_model(model)
    return quantize_apply(
        annotated_model, scheme=scheme, quantized_layer_name_prefix="quant_")
