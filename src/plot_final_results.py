# Project: Bachelor Thesis: Automated Quantization of Neural Networks
# Author: Miroslav Safar (xsafar23@stud.fit.vutbr.cz)

import argparse
import os

import matplotlib.pyplot as plt
import json
from matplotlib.ticker import FuncFormatter

from visualize.visualize_lib import load_data


def percent(x, pos=0):
    """Format function for matplotlib tickers"""
    return '%.0f %%' % (100 * x)


def main(configurations):
    """
    Creates graphs defined in final_graphs_configuration section of configurations file

    :param configurations: Configurations file
    """
    figure_definitions = json.load(open(configurations))["final_graphs_configuration"]

    for fig_def in figure_definitions:
        for i in range(2):
            en = i == 1
            fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(10, 6))
            (ax1) = axes

            ax1.set_title(fig_def["title_en"] if en else fig_def["title"])

            if en:
                ax1.set_xlabel("Size of weights in comparison to 8bit model [%]")
                ax1.set_ylabel("Top-1 relative accuracy to original model [%]")
            else:
                ax1.set_xlabel("Velikost vah v porovnání s 8-bitovým modelem [%]")
                ax1.set_ylabel("Relativní Top-1 přesnost vůči původnímu modelu [%]")

            memory_8bit_ref = fig_def["base_memory"]
            accuracy_ref = fig_def["base_accuracy"]

            for data_def in fig_def["data"]:
                data = load_data(os.path.join("../nsga_runs", data_def["source"]), pareto_filter=True)
                label = data_def["label_en"] if en else data_def["label"]

                ax1.step([x["memory"] / memory_8bit_ref for x in data], [x["accuracy"] / accuracy_ref for x in data],
                         color=data_def["color"],
                         label=label, where="post", marker=data_def["marker"], markersize=5, linewidth=0.5,
                         linestyle=data_def["line_style"])

                if len(data) > 0 and "accuracy_before" in data[0] and (
                        "include_before" not in data_def or data_def["include_before"]):
                    label_bef = (data_def["label_en"] + " (before fine-tuning)") if en else (
                            data_def["label"] + " (před dotrénováním)")
                    ax1.step([x["memory"] / memory_8bit_ref for x in data],
                             [x["accuracy_before"] / accuracy_ref for x in data],
                             color=data_def["color"],
                             label=label_bef, where="post", marker=data_def["marker"],
                             markersize=5, linewidth=0.5,
                             linestyle=data_def["line_style"], alpha=0.25)

            ax1.set_ylim(0, 1.2)
            ax1.set_xlim(0, 1)

            ax1.xaxis.set_major_formatter(FuncFormatter(percent))
            ax1.yaxis.set_major_formatter(FuncFormatter(percent))

            ax1.legend()

            # Ensure fig directory exists
            if not os.path.exists("fig"):
                os.makedirs("fig")

            fig.savefig(f'fig/{fig_def["output_file"]}{"_en" if en else ""}.png', dpi=300)
            fig.savefig(f'fig/{fig_def["output_file"]}{"_en" if en else ""}.pdf')

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='plot_final_results',
        description='Plot results of NSGA-II',
        epilog='')

    parser.add_argument('--configurations', default="graph_configurations.json")

    args = parser.parse_args()

    main(configurations=args.configurations)
