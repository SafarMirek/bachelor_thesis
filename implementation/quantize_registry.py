from abc import ABC, abstractmethod

import keras.layers


class QuantizeRegistry(ABC):

    def __init__(self):
        self.layer_config_map = []

    def get_quantize_config(self, layer: keras.layers.Layer, num_weight_bits: int, num_activations_bits: int):
        pass

    def is_quantizable(self, layer: keras.layers.Layer):
        pass
