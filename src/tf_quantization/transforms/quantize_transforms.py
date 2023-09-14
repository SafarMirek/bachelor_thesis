# Project: Bachelor Thesis: Automated Quantization of Neural Networks
# Author: Miroslav Safar (xsafar23@stud.fit.vutbr.cz)

from tensorflow import keras
import re

from tensorflow_model_optimization.python.core.quantization.keras import quantize_layout_transform
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit import \
    default_n_bit_transforms

from tf_quantization.transforms import custom_n_bit_quantize_layout_transform


class PerLayerQuantizeLayoutTransform(quantize_layout_transform.QuantizeLayoutTransform):
    """
        Ensures quantization is not places between layers that will be fused
    """

    def __init__(self, quantize_config, approx, per_channel, symmetric):
        self._quantize_config = quantize_config
        self._approx = approx
        self._per_channel = per_channel
        self._symmetric = symmetric

    def apply(self, model, layer_quantize_map):
        """
        Creates PerLayerQuantizeModelTransformer for model and makes transform
        :param model: Model
        :param layer_quantize_map: Map of layers metadata
        :return: transformed model
        """
        return PerLayerQuantizeModelTransformer(model, self._quantize_config, layer_quantize_map,
                                                self._approx, self._per_channel, self._symmetric).transform()


class PerLayerQuantizeModelTransformer:
    """
    Splits model layers into quantizable groups, that can have different quantization config
    and original model transformer can be used independently for each group instead for a whole model
    """
    _layers_groups = []
    _layers_group_index_map = {}

    def __init__(self, model, quantize_config, layer_quantize_map, approx, per_channel, symmetric):
        if not self._is_sequential_or_functional_model(model):
            raise ValueError(
                'Only tf.keras sequential or functional models can be transformed.')

        self.model = model
        self._quantize_config = quantize_config
        self._layer_quantize_map = layer_quantize_map
        self._approx = approx
        self._per_channel = per_channel
        self._symmetric = symmetric
        self._do_quantization_split()

    @staticmethod
    def _is_sequential_or_functional_model(model):
        """
        From: https://github.com/tensorflow/model-optimization
        """
        return PerLayerQuantizeModelTransformer._is_functional_model(model) or isinstance(
            model, keras.Sequential)

    @staticmethod
    def _is_functional_model(model):
        """
        From: https://github.com/tensorflow/model-optimization
        """
        return isinstance(model, keras.Model) \
            and not isinstance(model, keras.Sequential) \
            and model._is_graph_network  # pylint: disable=protected-access

    def _do_quantization_split(self):
        """
        This functions splits model into groups, that needs to be quantized together because of layer folding
        :return: Returns list that assigns for each layer a group and also sets _layers_groups instance variable
        """
        self._config = self.model.get_config()

        # We care only about layer patterns and they are same for all precisions
        self._transforms = [
            default_n_bit_transforms.InputLayerQuantize(),
            default_n_bit_transforms.SeparableConv1DQuantize(),
            default_n_bit_transforms.SeparableConvQuantize(),
            default_n_bit_transforms.Conv2DReshapeBatchNormReLUQuantize(),
            default_n_bit_transforms.Conv2DReshapeBatchNormActivationQuantize(),
            default_n_bit_transforms.Conv2DBatchNormReLUQuantize(),
            default_n_bit_transforms.Conv2DBatchNormActivationQuantize(),
            default_n_bit_transforms.Conv2DReshapeBatchNormQuantize(),
            default_n_bit_transforms.Conv2DBatchNormQuantize(),
            default_n_bit_transforms.ConcatTransform6Inputs(),
            default_n_bit_transforms.ConcatTransform5Inputs(),
            default_n_bit_transforms.ConcatTransform4Inputs(),
            default_n_bit_transforms.ConcatTransform3Inputs(),
            default_n_bit_transforms.ConcatTransform(),
            default_n_bit_transforms.LayerReLUQuantize(),
            default_n_bit_transforms.LayerReluActivationQuantize(),
            default_n_bit_transforms.DenseBatchNormQuantize(),
            default_n_bit_transforms.DenseBatchNormReLUQuantize(),
            default_n_bit_transforms.DenseBatchNormActivationQuantize(),
            custom_n_bit_quantize_layout_transform.Conv2DBatchNormReluQuantize(),
            custom_n_bit_quantize_layout_transform.DepthwiseConv2DBatchNormReluQuantize(),
            custom_n_bit_quantize_layout_transform.Conv2DBatchNormQuantize(),
        ]

        self._nodes = [layer["config"]["name"] for layer in self._config["layers"]]
        self._edges = {}
        for layer_name in self._nodes:
            self._edges[layer_name] = set()

        for transform in self._transforms:
            for layer_name in self._config["layers"]:
                group = self._match_layer_with_inputs(layer_name, transform.pattern())
                if group is None:
                    continue

                group_len = len(group)
                for i in range(0, group_len - 1):
                    node1 = group[i]
                    node2 = group[i + 1]
                    self._edges[node1].add(node2)
                    self._edges[node2].add(node1)

        # Get sub-graphs
        self._layers_groups = self._get_subgraphs(self._nodes, self._edges)
        self._create_layers_quantize_group_map()
        return self._layers_groups

    @staticmethod
    def _get_subgraphs(nodes, edges):
        """
        Get all graph components using BFS algorithm
        :param nodes: Nodes
        :param edges: Edges between nodes
        :return: sorted list of graph components
        """
        graphs = []
        remaining = set(nodes)

        while len(remaining) > 0:
            group = list()
            nodes = {remaining.pop()}
            while len(nodes) > 0:
                node = nodes.pop()
                for child in edges[node]:
                    if child in group:
                        continue
                    remaining.remove(child)
                    nodes.add(child)
                group.append(node)
                group = sorted(group)
            graphs.append(group)
        return sorted(graphs)

    @staticmethod
    def _match_pattern(target, pattern):
        """
        From: https://github.com/tensorflow/model-optimization
        """
        return re.match('^' + pattern + '$', target) is not None

    def _match_layer(self, layer, pattern):
        """
        Check if specific layer matches the pattern.
        From: https://github.com/tensorflow/model-optimization
        """
        if not self._match_pattern(layer['class_name'], pattern.class_name):
            return False

        layer_config = layer['config']
        for key, value in pattern.config.items():
            # Either the provided value should equal the config value, or
            # be a regex match to str(value).
            if not (self._match_pattern(str(layer_config.get(key)), str(value)) or layer_config.get(key) == value):
                return False

        return True

    def _match_layer_with_inputs(self, layer, pattern):
        """
        Match pattern at this layer, and continue to match at its inputs.
        From: https://github.com/tensorflow/model-optimization
        """

        if not self._match_layer(layer, pattern):
            return None

        if len(pattern.inputs) == 0:
            # Leaf layer in pattern.
            return [layer['config']['name']]

        # There is a possible edge case where a single layer may output multiple
        # tensors and multiple tensors from that layer may be used by the
        # connection. Ignoring those for now.
        input_layer_names = self._get_input_layer_names(layer)
        input_layers = self._get_layers(input_layer_names)

        if len(input_layers) != len(pattern.inputs):
            # Number of inputs this layer takes is different from the number of
            # inputs in the pattern.
            return None

        # Inbound layers can have different order from the list of input patterns.
        # Fix by checking all permutations.
        input_match_layer_nodes = []
        for input_layer, pattern_ in zip(input_layers, pattern.inputs):
            match_layer_node = self._match_layer_with_inputs(
                input_layer, pattern_)
            if not match_layer_node:
                return None
            input_match_layer_nodes = input_match_layer_nodes + match_layer_node

        return input_match_layer_nodes + [layer['config']['name']]

    def _get_input_layer_names(self, layer):
        """
        Get the names of a layer's input layers.
        From: https://github.com/tensorflow/model-optimization
        """
        if self._is_functional_model(self.model):
            inbound_nodes = layer['inbound_nodes']
            return [connection_info[0] for connection_info in inbound_nodes[0]]
        else:  # Sequential model.
            layers = self._config['layers']
            i = layers.index(layer)
            if i == 0:
                # First layer has no inputs.
                return []
            else:
                return [layers[i - 1]['config']['name']]

    def _get_layers(self, layer_names):
        return [
            layer for layer in self._config['layers']
            if layer['config']['name'] in layer_names
        ]

    def _find_pattern(self, pattern, matched_layers=None):
        """
        From: https://github.com/tensorflow/model-optimization
        """
        for layer in self._config['layers']:
            if matched_layers and layer['config']['name'] in matched_layers:
                continue
            match_layer = self._match_layer_with_inputs(layer, pattern)
            if match_layer:
                return match_layer

        return None

    def _get_layer_names(self, layer_node):
        """
        From: https://github.com/tensorflow/model-optimization
        """
        result = [layer_node.layer['config']['name']]
        for input_layer in layer_node.input_layers:
            result.extend(self._get_layer_names(input_layer))
        return result

    def _create_layers_quantize_group_map(self):
        self._layers_group_index_map = {}
        for i, group in enumerate(self._layers_groups):
            for layer_name in group:
                self._layers_group_index_map[layer_name] = i

    def transform(self):
        transformed_model = self.model
        layer_quantize_map = self._layer_quantize_map

        for i, group in enumerate(self._layers_groups):
            quantize_transform = custom_n_bit_quantize_layout_transform.CustomNBitQuantizeLayoutTransform(
                group,
                num_bits_weight=self._quantize_config[i]["weight_bits"],
                num_bits_activation=self._quantize_config[i]["activation_bits"],
                approx=self._approx,
                per_channel=self._per_channel,
                symmetric=self._symmetric
            )

            # layer_quantize_map gets modified by the transformations.
            transformed_model, layer_quantize_map = quantize_transform.apply(
                transformed_model, layer_quantize_map)

        return transformed_model, layer_quantize_map

    def get_number_of_quantizable_layers(self) -> int:
        """
        Returns number of quantizable layers (groups)
        :return: Number of quantizable layers (groups)
        """
        return len(self._layers_groups)

    def get_quantizable_layers_groups(self):
        """
        Return quantizable groups
        :return: List of groups
        """
        return self._layers_groups

    def get_layers_quantize_group_map(self):
        """
        Returns map that maps layers to their group
        :return: Map
        """
        return self._layers_group_index_map
