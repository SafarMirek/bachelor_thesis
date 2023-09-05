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


def get_label_by_param(language, param):
    if language == "en":
        if param == "accuracy":
            return "Top-1 accuracy [%]"
        if param == "total_energy":
            return "Energy [uJ]"
        if param == "total_cycles":
            return "Cycles [-]"
    else:
        if param == "accuracy":
            return "Top-1 accuracy [%]"
        if param == "total_energy":
            return "Energy [uJ]"
        if param == "total_cycles":
            return "Cycles [-]"
        if param == "total_edp":
            return "EDP [-]"


def main(configurations):
    """
    Creates graphs defined in final_graphs_configuration section of configurations file

    :param configurations: Configurations file
    """
    figure_definitions = json.load(open(configurations))["final_graphs_configuration"]

    param_combinations = [
        ("total_edp", "accuracy")
    ]

    for fig_def in figure_definitions:
        for i in range(1):
            en = i == 0
            fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(9, 12))
            (ax1) = axes

            ax1.set_title(fig_def["title_en"] if en else fig_def["title"])

            for c in range(3):
                x_param = param_combinations[c][0]
                y_param = param_combinations[c][1]

                ax1 = [ax1][c]

                ax1.set_xlabel(get_label_by_param("en" if en else "cz", x_param))
                ax1.set_ylabel(get_label_by_param("en" if en else "cz", y_param))

                # memory_8bit_ref = fig_def["base_memory"]
                accuracy_ref = fig_def["base_accuracy"]

                for data_def in fig_def["data"]:
                    data = load_data(os.path.join("../nsga_runs", data_def["source"]), pareto_filter=True,
                                     eval_prefix=data_def["eval_prefix"], sort_by=x_param)
                    label = data_def["label_en"] if en else data_def["label"]

                    ax1.step([x[x_param] for x in data], [x[y_param] for x in data],
                             color=data_def["color"],
                             label=label, where="post", marker=data_def["marker"], markersize=5, linewidth=0.5,
                             linestyle=data_def["line_style"])

                    if len(data) > 0 and f"{x_param}_before" in data[0] and (
                            "include_before" not in data_def or data_def["include_before"]):
                        label_bef = (data_def["label_en"] + " (before fine-tuning)") if en else (
                                data_def["label"] + " (před dotrénováním)")
                        ax1.step([x[x_param] for x in data],
                                 [x[f"{y_param}_before"] / accuracy_ref for x in data],
                                 color=data_def["color"],
                                 label=label_bef, where="post", marker=data_def["marker"],
                                 markersize=5, linewidth=0.5,
                                 linestyle=data_def["line_style"], alpha=0.25)

                # ax1.set_ylim(0, 1.2)
                # ax1.set_xlim(0, 1)

                # ax1.xaxis.set_major_formatter(FuncFormatter(percent))
                # ax1.yaxis.set_major_formatter(FuncFormatter(percent))

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
