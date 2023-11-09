# Project: Bachelor Thesis: Automated Quantization of Neural Networks
# Author: Miroslav Safar (xsafar23@stud.fit.vutbr.cz)

import glob
import gzip
import json
import os

from paretoarchive import PyBspTreeArchive


def load_data(run_dir, pareto_filter=False, pareto_x="memory", pareto_y="accuracy"):
    """
    Finds run evaluation and returns its data
    :param run_dir: Directory to NSGA run
    :param pareto_filter: Apply pareto filter and return only best pareto front
    :param pareto_x: objective 1
    :param pareto_y: objective 2
    :return: evaluated results of NSGA run
    """
    fn = sorted(glob.glob(os.path.join(run_dir, "eval.*.json.gz")))
    if len(fn) == 0:
        return []

    latest_fn = fn[-1]
    data = json.load(gzip.open(latest_fn))

    orig_fn = latest_fn.replace("eval.", "")
    try:
        orig_data = json.load(gzip.open(orig_fn))
        orig_data = orig_data["parent"] + orig_data["offspring"]
    except:
        orig_data = []

    if pareto_filter:
        data = apply_pareto_filter(data, pareto_x, pareto_y)

    for record in data:
        before_finetuning = list(filter(lambda x: x["quant_conf"] == record["quant_conf"], orig_data))
        if len(before_finetuning) > 0:
            record["accuracy_before"] = before_finetuning[0]["accuracy"]
            record["accuracy_max"] = max(record["accuracy"], record["accuracy_before"])
        else:
            record["accuracy_max"] = record["accuracy"]

    return sorted(data, key=lambda x: x[pareto_x])


def apply_pareto_filter(data, pareto_x="memory", pareto_y="accuracy"):
    """
    Returns only best Pareto-optimal set from provided data

    :param data: Data which should be filtered
    :param pareto_x: First optimization parameter
    :param pareto_y: Second optimization parameter
    :return: List of Pareto-optimal set ordered by first optimization parameter
    """
    pareto_data = [data[x] for x in
                   PyBspTreeArchive(2, minimizeObjective2=False).filter([(i[pareto_x], i[pareto_y]) for i in data],
                                                                        returnIds=True)]

    return sorted(pareto_data, key=lambda x: x[pareto_x])
