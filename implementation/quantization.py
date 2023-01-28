from tensorflow import keras
import tensorflow_model_optimization as tfmot


class BatchNormQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def __init__(self):
        pass

    # Configure weights to quantize with n-bit instead of 8-bits.
    # LastValueQuantizer - Quantize tensor based on range the last batch of values.
    def get_weights_and_quantizers(self, layer):
        return []

    # Configure weights to quantize with n-bit instead of 8-bits.
    # MovingAverageQuantizer - Quantize tensor based on a moving average of values across batches.
    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        # Add this line for each item returned in `get_weights_and_quantizers`
        # , in the same order
        pass

    def set_quantize_activations(self, layer, quantize_activations):
        # Add this line for each item returned in `get_activations_and_quantizers`
        # , in the same order.
        pass

    # Configure how to quantize outputs (may be equivalent to activations).
    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        return {}


class QuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def __init__(self, weight_bits, activation_bits):
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits

    # Configure weights to quantize with n-bit instead of 8-bits.
    # LastValueQuantizer - Quantize tensor based on range the last batch of values.
    def get_weights_and_quantizers(self, layer):
        return [(layer.kernel,
                 tfmot.quantization.keras.quantizers.LastValueQuantizer(num_bits=self.weight_bits, symmetric=True,
                                                                        narrow_range=False, per_axis=False))]

    # Configure weights to quantize with n-bit instead of 8-bits.
    # MovingAverageQuantizer - Quantize tensor based on a moving average of values across batches.
    def get_activations_and_quantizers(self, layer):
        return [(layer.activation,
                 tfmot.quantization.keras.quantizers.MovingAverageQuantizer(num_bits=self.activation_bits,
                                                                            symmetric=True, narrow_range=False,
                                                                            per_axis=False))]

    def set_quantize_weights(self, layer, quantize_weights):
        # Add this line for each item returned in `get_weights_and_quantizers`
        # , in the same order
        layer.kernel = quantize_weights[0]

    def set_quantize_activations(self, layer, quantize_activations):
        # Add this line for each item returned in `get_activations_and_quantizers`
        # , in the same order.
        layer.activation = quantize_activations[0]

    # Configure how to quantize outputs (may be equivalent to activations).
    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        return {"weight_bits": self.weight_bits, "activation_bits": self.activation_bits}


def __get_quantize_config(layer, num_weight_bits: int, num_activations_bits: int):
    if isinstance(layer, keras.layers.BatchNormalization):
        return BatchNormQuantizeConfig()
    return QuantizeConfig(num_weight_bits, num_activations_bits)


def create_quantization_function(quantization_model: list[int]):
    i = 0

    quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer

    def clone_function(layer):
        nonlocal i
        if not __is_quantizable(layer):
            return layer
        if i < len(quantization_model):
            num_bits = quantization_model[i]
            i += 1
        else:
            num_bits = 8
        return quantize_annotate_layer(layer, quantize_config=__get_quantize_config(layer, num_bits, 8))

    return clone_function


def __is_quantizable(layer):
    return (isinstance(layer, keras.layers.Dense) or
            isinstance(layer, keras.layers.Conv2D) or
            isinstance(layer, keras.layers.DepthwiseConv2D) or
            isinstance(layer, keras.layers.BatchNormalization)
            )


def number_of_quantizable_layers(model):
    count = 0
    for layer in model.layers:
        if __is_quantizable(layer):
            count = count + 1
    return count


def quantize_model(model, quantization_model: list[int]):
    clone = keras.models.clone_model(model, clone_function=create_quantization_function(quantization_model))
    return tfmot.quantization.keras.quantize_annotate_model(clone)


def apply_custom_quantization(annotated_model):
    with tfmot.quantization.keras.quantize_scope(
            {
                'QuantizeConfig': QuantizeConfig,
                'BatchNormQuantizeConfig': BatchNormQuantizeConfig
            }
    ):
        # Use `quantize_apply` to actually make the model quantization aware.
        quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
        return quant_aware_model
