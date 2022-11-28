from tensorflow import keras
import tensorflow_model_optimization as tfmot


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


def __get_quantize_config(num_weight_bits: int, num_activations_bits: int):
    return QuantizeConfig(num_weight_bits, num_activations_bits)


def create_quantization_function(quantization_model: list[int]):
    i = 0

    def clone_function(layer):
        nonlocal i
        if i < len(quantization_model):
            if not isinstance(layer, keras.layers.Dense) and not isinstance(layer, keras.layers.Conv2D):
                return layer
            num_bits = quantization_model[i]
            i += 1
            return tfmot.quantization.keras.quantize_annotate_layer(layer,
                                                                    quantize_config=__get_quantize_config(num_bits, 8))
        return layer

    return clone_function


def quantize_model(model, quantization_model: list[int]):
    clone = keras.models.clone_model(model, clone_function=create_quantization_function(quantization_model))
    return tfmot.quantization.keras.quantize_annotate_model(clone)


def apply_custom_quantization(annotated_model):
    with tfmot.quantization.keras.quantize_scope(
            {
                'QuantizeConfig': QuantizeConfig
            }
    ):
        # Use `quantize_apply` to actually make the model quantization aware.
        quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
        return quant_aware_model
