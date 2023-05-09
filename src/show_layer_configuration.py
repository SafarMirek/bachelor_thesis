# Project: Bachelor Thesis: Automated Quantization of Neural Networks
# Author: Miroslav Safar (xsafar23@stud.fit.vutbr.cz)

import argparse
import os.path

import inquirer
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from tensorflow import keras
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapperV2

from tf_quantization.quantize_model import quantize_model
from tf_quantization.transforms.quantize_transforms import PerLayerQuantizeModelTransformer
import calculate_model_size_lib
from visualize.visualize_lib import load_data


def _get_mask(base_model, per_channel, symmetric):
    """
    Create mask that maps chromosome to all layers configuration

    :return: created mask
    """
    transformer = PerLayerQuantizeModelTransformer(base_model, [], {}, approx=True,
                                                   per_channel=per_channel, symmetric=symmetric)

    groups = transformer.get_quantizable_layers_groups()
    mask = [-1 for _ in range(len(groups))]
    count = 0
    for i, layers_group in enumerate(groups):
        if calculate_model_size_lib.calculate_weights_mobilenet_size(base_model, only_layers=layers_group,
                                                                     per_channel=per_channel,
                                                                     symmetric=symmetric) > 0:
            mask[i] = count
            count = count + 1

    return mask


def _apply_mask(mask, masked_quant_config):
    """
    Converts masked quantization configuration to configuration for all model layers

    :param mask: Used mask
    :param masked_quant_config: Masked quantization configuration
    :return: configuration for all model layers
    """
    masked_quant_config.append(8)
    final_quant_config = [masked_quant_config[i] for i in mask]
    config = [
        {
            "weight_bits": final_quant_config[i],
            "activation_bits": 8
        } for i in range(len(mask))
    ]
    return config


def get_layers_data(base_model_path, per_channel, symmetric, masked_quant_config_list):
    """
    Get layers bit-width for model quantized by provided masked configuration

    :param base_model_path: Path to base model
    :param per_channel: per-channel/per-tensor quantization
    :param symmetric: symmetric/asymmetric quantization
    :param masked_quant_config_list: List of masked quantization configurations
    :return: List of layers and weight bit-width for each layer for all input masked quantization configurations
    """
    base_model = keras.models.load_model(base_model_path)

    mask = _get_mask(base_model, per_channel, symmetric)

    layers = get_layers(base_model, _apply_mask(mask, [8 for _ in range(len(masked_quant_config_list[0]))]),
                        per_channel=per_channel, symmetric=symmetric)

    ordered_quant_config_list = []
    for masked_quant_config in masked_quant_config_list:
        quantize_config = _apply_mask(mask, masked_quant_config)
        ordered_quant_config_list.append(
            get_layers_with_quant_conf(base_model, per_channel, symmetric, quantize_config, layers)
        )

    return layers, ordered_quant_config_list


def get_layers(base_model, quantize_config, per_channel, symmetric):
    """
    Get list of layers of quantized model

    :param base_model: Base model
    :param quantize_config: Quantize config
    :param per_channel: per-channel/per-tensor quantization
    :param symmetric: symmetric/asymmetric quantization
    :return: List of model layers
    """
    q_model = quantize_model(base_model, quantize_config, approx=True, per_channel=per_channel, symmetric=symmetric)
    output = []
    for layer in q_model.layers:
        if calculate_model_size_lib.calculate_weights_mobilenet_size(q_model, per_channel=per_channel,
                                                                     symmetric=symmetric,
                                                                     only_layers=[layer.name]) > 0:
            output.append(layer.name)

    return output


def get_layers_with_quant_conf(base_model, per_channel, symmetric, quantize_config, layers):
    """
    Get bit-width for specified layers in model quantized by specified quantization configuration

    :param base_model: Base model
    :param per_channel: per-channel/per-tensor quantization
    :param symmetric: symmetric/asymmetric quantization
    :param quantize_config: Configuration of quantization for all quantizable groups
    :param layers: Layers we are interested in
    :return: List of bit-width configuration for provided layers
    """
    q_model = quantize_model(base_model, quantize_config, approx=True, per_channel=per_channel, symmetric=symmetric)

    output = []
    for layer in q_model.layers:
        if layer.name in layers:
            num_bits_weight = 8
            if hasattr(layer, "quantize_num_bits_weight"):
                num_bits_weight = layer.quantize_num_bits_weight
            elif isinstance(layer, QuantizeWrapperV2):
                if "num_bits_weight" in layer.quantize_config.get_config():
                    num_bits_weight = layer.quantize_config.get_config()["num_bits_weight"]

            output.append(num_bits_weight)
    return output


def show_solo_graph(selected_quant_conf, per_channel, symmetric, base_model_path):
    """
    Visualize layers with their bit-width of quantized model by provided quantization configuration

    :param selected_quant_conf: Quantization configuration
    :param per_channel: per-channel/per-tensor weights quantization
    :param symmetric: symmetric/asymmetric weights quantization
    :param base_model_path: Path to base model
    """
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(10, 5))
    (ax1) = axes

    ax1.set_title("Konfigurace kvantizace vah jednotlivých vrstev")

    ax1.set_xlabel("Jméno vrstvy")
    ax1.set_ylabel("Úroveň kvantizace vah [v bitech]")

    ax1.set_ylim(0, 8)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_position('zero')

    layers, [confs] = get_layers_data(base_model_path, per_channel=True, symmetric=True,
                                      masked_quant_config_list=[selected_quant_conf])

    ax1.grid(axis="y", color="black", alpha=.3, linewidth=.5, linestyle=":")
    ax1.bar(layers, confs, width=0.7, bottom=0, align='center', color='C3')

    ax1.set_xticklabels(layers, rotation=80)
    fig.tight_layout()

    # Ensure fig directory exists
    if not os.path.exists("fig"):
        os.makedirs("fig")

    fig.savefig(f"fig/conf_{'_'.join(str(e) for e in selected_quant_conf)}_{per_channel}_{symmetric}.pdf")
    fig.savefig(f"fig/conf_{'_'.join(str(e) for e in selected_quant_conf)}_{per_channel}_{symmetric}.png",
                dpi=300)

    plt.show()


def show_heatmap(base_model_path, per_channel, symmetric, data, run_name, base_accuracy):
    """
    This function creates heatmap of layer bit-width in quantized model by best found quantization configurations
    in the NSGA-II run

    :param base_model_path: Path to base model
    :param per_channel: per-channel/per-tensor weights quantization
    :param symmetric: symmetric/asymmetric weights quantization
    :param data: Run data
    :param run_name: Name of the run of NSGA-II for output file name
    :param base_accuracy: Base accuracy of model for calculating relative loss of accuracy
    """
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(12, 6))
    (ax1) = axes

    ax1.set_title("Konfigurace kvantizace vah jednotlivých vrstev")

    ax1.set_xlabel("Jméno vrstvy")
    ax1.set_ylabel("Konfigurace kvantizace")

    if len(data) > 8:
        data = data[-8:]  # Take only best 8

    layers, confs_list = get_layers_data(base_model_path, per_channel=per_channel, symmetric=symmetric,
                                         masked_quant_config_list=[conf["quant_conf"] for conf in data])

    # Get a list of colors from the colormap (we want only first 8 colors)
    cmap = plt.get_cmap('coolwarm')
    resampled_colors = np.linspace(0, 1, 20)
    resampled_colors = [cmap(x) for x in resampled_colors][0:7][::-1]

    new_cmap = ListedColormap(resampled_colors)
    im = ax1.imshow(confs_list[::-1], vmin=1.5, vmax=8.5, cmap=new_cmap)

    ax1.set_xticks(np.arange(len(layers)), labels=layers, rotation=80, ha="right")
    total_len = len(data)
    ax1.set_yticks(np.arange(len(confs_list)),
                   labels=[f"#{i + 1} ({(((conf['accuracy_max'] / base_accuracy) - 1) * 100):.2f}%)" for i, conf in
                           enumerate(data[::-1])])

    fig.colorbar(im, label='Bitová šířka', ax=ax1)

    for i in range(total_len):
        for j in range(len(layers)):
            text = ax1.text(j, i, confs_list[::-1][i][j],
                            ha="center", va="center", color="w")

    fig.tight_layout()

    # Ensure fig directory exists
    if not os.path.exists("fig"):
        os.makedirs("fig")

    fig.savefig(f"fig/best_conf_{run_name}.pdf")
    fig.savefig(f"fig/best_conf_{run_name}.png",
                dpi=300)

    plt.show()


def main(run, configuration, per_channel, symmetric, base_model_path, all, base_accuracy):
    """
    This script visualize layer bit-width assignment of NSGA-II results

    :param run: NSGA-II run folder
    :param configuration: Quantization configuration (alternative to NSGA-II run folder)
    :param per_channel: per-channel/per-tensor weights quantization
    :param symmetric: symmetric/asymmetric weights quantization
    :param base_model_path: Path to base model
    :param all: Visualize all best found solutions (we show best 8)
    :param base_accuracy: Base model accuracy for comparison
    """
    if run is not None:
        run_data = load_data(run, pareto_filter=True)

        if all:
            show_heatmap(base_model_path, per_channel, symmetric, run_data, os.path.basename(run), base_accuracy)
            return

        questions = [
            inquirer.List(
                'configuration',
                message="Select configuration",
                choices=[
                    f'{i}. Weights size: {(record["memory"] / 8000):.2f}kB Accuracy: {(record["accuracy_max"] * 100):.2f}%'
                    for i, record in enumerate(run_data, start=1)],
            ),
        ]
        answers = inquirer.prompt(questions)

        selected_quant_conf = run_data[int(answers["configuration"].split(".")[0]) - 1]["quant_conf"]
    else:
        selected_quant_conf = configuration

    show_solo_graph(selected_quant_conf, per_channel, symmetric, base_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='show_layer_configuration',
        description='Plot layer quantization configuration from NSGA chromosome',
        epilog='')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run")
    group.add_argument("--configuration")

    parser.add_argument('--per-channel', default=False, action='store_true')
    parser.add_argument('--symmetric', default=False, action='store_true')
    parser.add_argument('--all', default=False, action='store_true')

    parser.add_argument(
        '--base-model-path',
        type=str,
        default="mobilenet_tinyimagenet_025.keras",
        help='')

    parser.add_argument(
        '--base-accuracy',
        type=float,
        default=0.516,
        help='')

    args = parser.parse_args()

    main(run=args.run, configuration=args.configuration, per_channel=args.per_channel, symmetric=args.symmetric,
         base_model_path=args.base_model_path, all=args.all, base_accuracy=args.base_accuracy)
