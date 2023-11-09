# Project: Bachelor Thesis: Automated Quantization of Neural Networks
# Author: Miroslav Safar (xsafar23@stud.fit.vutbr.cz)

import argparse
import glob
import gzip
import json
import os
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from visualize.visualize_lib import apply_pareto_filter


def percent(x, pos=0):
    """Format function for matplotlib tickers"""
    return '%.0f %%' % (100 * x)


def main(*, noshow, all, approx, per_channel, symmetric, configurations, act_quant, en):
    """
    Plot best Pareto-optimal sets during NSGA-II for runs configured in generations_config section
    of provided configurations file

    :param noshow: Only save plots as files and do not show them
    :param all: Plot all configured runs
    :param approx: Filter runs which used approximate method for solution of bn folding
    :param per_channel: Filter per-channel/per-tensor weights quantization runs
    :param symmetric: Filter symmetric/asymmetric weights quantization runs
    :param configurations: Configurations file that contains generations_config section
    :param act_quant: Filter runs with enabled activation quantization
    :param en: English titles in graphs
    """
    configurations = json.load(open(configurations))["generations_config"]
    if not all:
        configurations = list(
            filter(
                lambda x: x["per_channel"] == per_channel and x["symmetric"] == symmetric and x[
                    "approx"] == approx and x["act_quant"] == act_quant,
                configurations
            )
        )

    if len(configurations) == 0:
        raise ValueError("No configuration for this setting was not found.")

    for configuration in configurations:
        title = configuration["title_en"] if en else configuration["title"]
        runs_folder = configuration["run_folder"]
        #memory_8bit_ref = configuration["base_memory"]
        ref_accuracy = configuration["float_accuracy"]
        alldata = {}

        for fn in sorted(glob.glob("../nsga_runs/%s/run.*.json.gz" % runs_folder)):
            gen = int(re.match(r".*run\.(\d+)\.json\.gz", fn).group(1))
            alldata[gen] = json.load(gzip.open(fn))
            for record in alldata[gen]["parent"] + alldata[gen]["offspring"]:
            #    record["memory_percent"] = record["memory"] / memory_8bit_ref
                record["accuracy_percent"] = record["accuracy"] / ref_accuracy

        selected_generations = [1] + configuration["generations"]
        colors = ["black", "tab:green", "tab:green", "#31a354", "#006d2c", "#005723", "#00411a", "#00210d"]
        colors_alpha = [1, 0.5, 1, 1, 1, 1, 1, 1]

        fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(8, 5))
        (ax1) = axes

        ax1.set_title(title)

        if args.en:
            ax1.set_xlabel("EDP [J*cycles]")
            ax1.set_ylabel("Top-1 relative accuracy after partly fine-tuning")
        else:
            ax1.set_xlabel("EDP [J*cycles]")
            ax1.set_ylabel("Top-1 relativní přesnost po částečném dotrénování")

        ax1.set_ylim(0, 1.05)
        #ax1.set_xlim(0, 700000)

        #ax1.xaxis.set_major_formatter(FuncFormatter(percent))
        ax1.yaxis.set_major_formatter(FuncFormatter(percent))

        for i, generation in enumerate(selected_generations):
            color = colors[i]
            color_alpha = colors_alpha[i]
            data = alldata[generation]["parent"]
            data = apply_pareto_filter(data, sort_by="total_edp")

            label = "Best configurations" if en else "Nejlepší konfigurace"

            ax1.step([x["total_edp"] for x in data], [x["accuracy_percent"] for x in data], color=color,
                     label=f"{label} ({generation}. gen)", where="post", marker="x" if i == 0 else "o", markersize=4,
                     linewidth=0.5,
                     linestyle=":", alpha=color_alpha)

        ax1.legend()
        output = configuration['out'] if 'out' in configuration else runs_folder

        # Ensure fig directory exists
        if not os.path.exists("fig"):
            os.makedirs("fig")

        fig.savefig(f"fig/{output}_generations{'_en' if en else ''}.pdf")
        fig.savefig(f"fig/{output}_generations{'_en' if en else ''}.png", dpi=300)

    if not noshow:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='plot_nsga_generations',
        description='Plot NSGA generations',
        epilog='')

    parser.add_argument('--noshow', default=False, action='store_true')
    parser.add_argument('--all', default=False, action='store_true')
    parser.add_argument('--approx', default=False, action='store_true')
    parser.add_argument('--per-channel', default=False, action='store_true')
    parser.add_argument('--symmetric', default=False, action='store_true')
    parser.add_argument('--configurations', default="graph_configurations.json")
    parser.add_argument('--act-quant', default=False, action='store_true')

    parser.add_argument('--en', default=False, action='store_true')

    args = parser.parse_args()

    main(noshow=args.noshow, all=args.all, approx=args.approx, per_channel=args.per_channel,
         symmetric=args.symmetric, configurations=args.configurations, act_quant=args.act_quant, en=True)
