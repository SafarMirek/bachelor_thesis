import warnings

import keras.layers
import tensorflow_model_optimization as tfmot
from keras import initializers
from tensorflow import keras
from tensorflow_model_optimization.python.core.quantization.keras import quantize_registry, quantizers, quant_ops
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit.default_n_bit_quantize_registry import \
    DefaultNBitQuantizeConfig
from tensorflow_model_optimization.python.core.quantization.keras.quantize_config import QuantizeConfig
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import _QuantizeHelper, Quantizer, \
    LastValueQuantizer

DefaultNBitQuantizeRegistry = tfmot.quantization.keras.experimental.default_n_bit.DefaultNBitQuantizeRegistry


class PerLayerNBitQuantizeRegistry(quantize_registry.QuantizeRegistry):
    """
    Quantize registry for per-layer quantization
    """

    def __init__(self, quantization_config, activation_quant_no_affect=False, per_channel=True, symmetric=True):
        """
        TODO:
        :param quantization_config: List of quantization configs for each layer
        :param activation_quant_no_affect: Disable activation quantization
        :param per_channel: Use per-channel quantization for weights of convolution layers
        :param symmetric: Use symmetric or asymmetric quantization for weights of convolution layers
        """
        self._quantization_config = quantization_config
        self.registries = [
            [
                DefaultNBitQuantizeRegistry(num_bits_weight=w + 1, num_bits_activation=a + 1)
                for a in range(8)
            ]
            for w in range(8)
        ]
        self.activation_quant_no_affect = activation_quant_no_affect
        self.per_channel = per_channel
        self.symmetric = symmetric

    def get_quantize_config(self, layer: keras.layers.Layer):
        """

        :param layer: layer to be quantized
        :return: QuantizeConfig for layer
        """
        num_weight_bits = 8
        num_activation_bits = 8

        if layer.name in self._quantization_config:
            num_weight_bits = self._quantization_config[layer.name]["weight_bits"]
            num_activation_bits = self._quantization_config[layer.name]["activation_bits"]
        else:
            warnings.warn(f'No configuration found for {layer.name}. Using 8 bit default quantization')

        if isinstance(layer, keras.layers.ReLU):
            return CustomNBitQuantizeConfig([], [], True,
                                            keras.initializers.Constant(0.0),
                                            keras.initializers.Constant(6.0),
                                            num_bits_weight=num_weight_bits,
                                            num_bits_activation=num_activation_bits,
                                            activation_quant_no_affect=self.activation_quant_no_affect
                                            )

        if isinstance(layer, keras.layers.Conv2D):
            return CustomNBitConvQuantizeConfig(
                min_initializer=keras.initializers.Constant(-6.0),
                max_initializer=keras.initializers.Constant(6.0),
                num_bits_weight=num_weight_bits,
                num_bits_activation=num_activation_bits,
                activation_quant_no_affect=self.activation_quant_no_affect,
                per_axis=self.per_channel,
                symmetric=self.symmetric
            )

        return self.registries[num_weight_bits - 1][num_activation_bits - 1].get_quantize_config(layer)

    def supports(self, layer: keras.layers.Layer):
        """
        Checks if registry supports given layer
        :param layer: layer to be quantized
        :return: true if quantize registry supports given layer
        """
        return self.registries[8 - 1][8 - 1].supports(layer)


class CustomNBitConvQuantizeConfig(DefaultNBitQuantizeConfig):
    def __init__(self, min_initializer, max_initializer, num_bits_weight: int = 8, num_bits_activation: int = 8,
                 activation_quant_no_affect=False, symmetric=True, per_axis=True):
        super().__init__(
            ['kernel'], ['activation'], False,
            num_bits_weight=num_bits_weight,
            num_bits_activation=num_bits_activation)
        self.weight_quantizer = LastValueQuantizer(
            num_bits=num_bits_weight,
            per_axis=per_axis,
            symmetric=symmetric,
            narrow_range=True
        )
        self.activation_quantizer = CustomMovingAverageQuantizer(
            num_bits=num_bits_activation, per_axis=False,
            symmetric=False, narrow_range=False, min_initializer=min_initializer, max_initializer=max_initializer,
            no_affect=activation_quant_no_affect)


class CustomNBitQuantizeConfig(QuantizeConfig):
    """QuantizeConfig for non recurrent Keras layers."""

    def __init__(self, weight_attrs, activation_attrs, quantize_output, min_initializer, max_initializer,
                 num_bits_weight: int = 8, num_bits_activation: int = 8, activation_quant_no_affect=False):
        self.weight_attrs = weight_attrs
        self.activation_attrs = activation_attrs
        self.quantize_output = quantize_output
        self._num_bits_weight = num_bits_weight
        self._num_bits_activation = num_bits_activation
        self._activation_quant_no_affect = activation_quant_no_affect

        # TODO(pulkitb): For some layers such as Conv2D, per_axis should be True.
        # Add mapping for which layers support per_axis.
        self.weight_quantizer = quantizers.LastValueQuantizer(
            num_bits=num_bits_weight, per_axis=False,
            symmetric=True, narrow_range=True)  # weight
        self.activation_quantizer = CustomMovingAverageQuantizer(
            num_bits=num_bits_activation, per_axis=False,
            symmetric=False, narrow_range=False, min_initializer=min_initializer,
            max_initializer=max_initializer, no_affect=activation_quant_no_affect)  # activation/output

    def get_weights_and_quantizers(self, layer):
        return [(getattr(layer, weight_attr), self.weight_quantizer)
                for weight_attr in self.weight_attrs]

    def get_activations_and_quantizers(self, layer):
        return [(getattr(layer, activation_attr), self.activation_quantizer)
                for activation_attr in self.activation_attrs]

    def set_quantize_weights(self, layer, quantize_weights):
        if len(self.weight_attrs) != len(quantize_weights):
            raise ValueError(
                '`set_quantize_weights` called on layer {} with {} '
                'weight parameters, but layer expects {} values.'.format(
                    layer.name, len(quantize_weights), len(self.weight_attrs)))

        for weight_attr, weight in zip(self.weight_attrs, quantize_weights):
            current_weight = getattr(layer, weight_attr)
            if current_weight.shape != weight.shape:
                raise ValueError('Existing layer weight shape {} is incompatible with'
                                 'provided weight shape {}'.format(
                    current_weight.shape, weight.shape))

            setattr(layer, weight_attr, weight)

    def set_quantize_activations(self, layer, quantize_activations):
        if len(self.activation_attrs) != len(quantize_activations):
            raise ValueError(
                '`set_quantize_activations` called on layer {} with {} '
                'activation parameters, but layer expects {} values.'.format(
                    layer.name, len(quantize_activations),
                    len(self.activation_attrs)))

        for activation_attr, activation in zip(
                self.activation_attrs, quantize_activations):
            setattr(layer, activation_attr, activation)

    def get_output_quantizers(self, layer):
        if self.quantize_output:
            return [self.activation_quantizer]
        return []

    @classmethod
    def from_config(cls, config):
        """Instantiates a `DefaultNBitQuantizeConfig` from its config.

        Args:
            config: Output of `get_config()`.

        Returns:
            A `DefaultNBitQuantizeConfig` instance.
        """
        return cls(**config)

    def get_config(self):
        # TODO(pulkitb): Add weight and activation quantizer to config.
        # Currently it's created internally, but ideally the quantizers should be
        # part of the constructor and passed in from the registry.
        return {
            'weight_attrs': self.weight_attrs,
            'activation_attrs': self.activation_attrs,
            'quantize_output': self.quantize_output,
            'num_bits_weight': self._num_bits_weight,
            'num_bits_activation': self._num_bits_activation,
            "activation_quant_no_affect": self._activation_quant_no_affect
        }


class CustomMovingAverageQuantizer(_QuantizeHelper, Quantizer):
    """Quantize tensor based on a moving average of values across batches."""

    def __init__(self, num_bits, per_axis, symmetric, narrow_range, min_initializer, max_initializer, no_affect=False):
        """Construct a MovingAverageQuantizer.

        This is an experimental API not subject to backward compatibility.

        Args:
          num_bits: Number of bits for quantization
          per_axis: Whether to apply per_axis quantization. The last dimension is
            used as the axis.
          symmetric: If true, use symmetric quantization limits instead of training
            the minimum and maximum of each quantization range separately.
          narrow_range: In case of 8 bits, narrow_range nudges the quantized range
            to be [-127, 127] instead of [-128, 127]. This ensures symmetric
            range has 0 as the centre.
        """
        self.num_bits = num_bits
        self.per_axis = per_axis
        self.symmetric = symmetric
        self.narrow_range = narrow_range
        self.min_initializer = initializers.get(min_initializer)
        self.max_initializer = initializers.get(max_initializer)
        self.no_affect = no_affect

    def build(self, tensor_shape, name, layer):
        shape = None
        if self.per_axis and tensor_shape is not None:
            shape = (tensor_shape[-1])

        min_weight = layer.add_weight(
            name + '_min',
            initializer=self.min_initializer,
            trainable=False,
            shape=shape)
        max_weight = layer.add_weight(
            name + '_max',
            initializer=self.max_initializer,
            trainable=False,
            shape=shape)

        return {'min_var': min_weight, 'max_var': max_weight}

    def __call__(self, inputs, training, weights, **kwargs):
        """Quantize tensor.

        Args:
          inputs: Input tensor to be quantized.
          training: Whether the graph is currently training.
          weights: Dictionary of weights the quantizer can use to quantize the
            tensor. This contains the weights created in the `build` function.
          **kwargs: Additional variables which may be passed to the quantizer.

        Returns:
          Quantized tensor.
        """
        quant_inputs = quant_ops.MovingAvgQuantize(
            inputs,
            weights['min_var'],
            weights['max_var'],
            ema_decay=0.999,
            is_training=training,
            num_bits=self.num_bits,
            per_channel=self.per_axis,
            symmetric=self.symmetric,
            narrow_range=self.narrow_range,
        )
        if self.no_affect and training:
            return inputs
        else:
            return quant_inputs

    def get_config(self):
        return {
            'num_bits': self.num_bits,
            'per_axis': self.per_axis,
            'symmetric': self.symmetric,
            'narrow_range': self.narrow_range,
            "min_initializer": initializers.serialize(self.min_initializer),
            "max_initializer": initializers.serialize(self.max_initializer),
            "no_affect": self.no_affect
        }

    def __eq__(self, other):
        if not isinstance(other, CustomMovingAverageQuantizer):
            return False

        return (self.num_bits == other.num_bits and
                self.per_axis == other.per_axis and
                self.symmetric == other.symmetric and
                self.narrow_range == other.narrow_range)

    def __ne__(self, other):
        return not self.__eq__(other)
