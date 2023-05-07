# Project: Bachelor Thesis: Automated Quantization of Neural Networks
# Author: Miroslav Safar (xsafar23@stud.fit.vutbr.cz)
import argparse
import os

from matplotlib import pyplot as plt

import calculate_model_size_lib
from tf_quantization.quantize_model import quantize_model
from tf_quantization.transforms.custom_n_bit_quantize_layout_transform import keras


def main(base_model_path):
    """
    This scripts plots number of weights in each layer in provided model

    :param base_model_path: Path to base model
    """
    base_model = keras.models.load_model(base_model_path)

    quant_layer_conf = {"weight_bits": 8, "activation_bits": 8}
    q_model = quantize_model(base_model, [quant_layer_conf for _ in range(37)], approx=True, per_channel=True,
                             symmetric=True)

    layers_with_weights_number = calculate_model_size_lib.calculate_mobilenet_per_layer_number_of_weights(q_model)
    layers_with_weights_number = list(filter(lambda x: x[1] > 0, layers_with_weights_number))

    show_graph(layers_with_weights_number)


def show_graph(layers_with_weights_number):
    """
    This method create graph from list of layers with number of weights in them
    """
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(11, 5))
    (ax1) = axes

    ax1.set_title("Počet kvantizovatelných vah v jednotlivých vrstvách")

    ax1.set_xlabel("Jméno vrstvy")
    ax1.set_ylabel("Počet vah")

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_position('zero')

    layers = [layer[0] for layer in layers_with_weights_number]
    weights_number = [layer[1] for layer in layers_with_weights_number]

    colors = ["lightskyblue" if "dw" in layer[0] else "dodgerblue" for layer in layers_with_weights_number]

    total_number_of_weights = sum(weights_number)

    ax1.grid(axis="y", color="black", alpha=.3, linewidth=.5, linestyle=":")
    bars = ax1.bar(layers, weights_number, width=0.7, bottom=0, align='center', color=colors)

    for bar in bars:
        ax1.annotate(f'{bar.get_height() * 100 / total_number_of_weights:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=9)

    ax1.set_xticklabels(layers, rotation=80)
    fig.tight_layout()

    # Ensure fig directory exists
    if not os.path.exists("fig"):
        os.makedirs("fig")

    fig.savefig(f"fig/mobilenet_025_weights_per_layer.pdf")
    fig.savefig(f"fig/mobilenet_025_weights_per_layer.png",
                dpi=300)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--base-model-path',
        type=str,
        default="mobilenet_tinyimagenet_025.keras",
        help='')

    args = parser.parse_args()

    main(base_model_path=args.base_model_path)
