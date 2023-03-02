import warnings
from abc import ABC, abstractmethod

import keras.layers

import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.quantization.keras import quantize_registry

DefaultNBitQuantizeRegistry = tfmot.quantization.keras.experimental.default_n_bit.DefaultNBitQuantizeRegistry


class QuantizeRegistry(quantize_registry.QuantizeRegistry):

    def __init__(self, quantization_config):
        self._quantization_config = quantization_config
        self.registries = [
            [
                DefaultNBitQuantizeRegistry(num_bits_weight=w + 1, num_bits_activation=a + 1)
                for a in range(8)
            ]
            for w in range(8)
        ]

    def get_quantize_config(self, layer: keras.layers.Layer):
        num_weight_bits = 8
        num_activation_bits = 8

        if layer.name in self._quantization_config:
            num_weight_bits = self._quantization_config[layer.name]["weight_bits"]
            num_activation_bits = self._quantization_config[layer.name]["activation_bits"]
        else:
            warnings.warn(f'No configuration found for {layer.name}. Using 8 bit default quantization')

        return self.registries[num_weight_bits - 1][num_activation_bits - 1].get_quantize_config(layer)

    def supports(self, layer: keras.layers.Layer):
        return self.registries[8 - 1][8 - 1].supports(layer)
