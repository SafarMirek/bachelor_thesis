from __future__ import print_function

import keras.models
from paretoarchive import PyBspTreeArchive
import json, gzip
import random
import datetime, os
import argparse
import tensorflow as tf

import calculate_model_size
from nsga_lib import crowding_reduce, Analyzer

parser = argparse.ArgumentParser()

parser.add_argument(
    '--logs-dir',
    type=str,
    default="/tmp/safarmirek/run" + datetime.datetime.now().strftime("-%Y%m%d-%H%M"),
    help='Logs dir')

parser.add_argument(
    '--previous-run',
    type=str,
    default="",
    help='')

parser.add_argument(
    '--base_model_path',
    type=str,
    default="",
    help='')

parser.add_argument(
    '--batch-size',
    type=int,
    default=1000,
    help='Number of images in the batch.')

parser.add_argument(
    '--iterations',
    type=int,
    default=10,
    help='Number of iterations of batches, batch_size * iterations <= data_size; batch_size >> iterations.')

parser.add_argument(
    '--generations',
    type=int,
    default=25,
    help='Number of images in the batch.')


def make_config(conf):
    return {"quant_conf": conf}


def main(*, logs_dir, base_model_path, quantizable_layers, parent_size=50, offspring_size=50, batch_size=64,
         qat_epochs=6,
         generations=25, pretrained_qat_weights_path=None):
    # Create folder for logs
    os.makedirs(logs_dir)

    base_model = keras.models.load_model(base_model_path)

    maximal = {
        "accuracy": 1.0,
        "memory": calculate_model_size.calculate_weights_mobilenet_size(base_model)
    }

    analyzer = Analyzer(base_model, batch_size=batch_size, qat_epochs=qat_epochs,
                        pretrained_qat_weights_path=pretrained_qat_weights_path)

    g_last = 0
    parents = [[i for _ in range(quantizable_layers)] for i in range(2, 9)]
    next_parent = list(analyzer.analyze([make_config(x) for x in parents]))

    offsprings = []

    for g in range(g_last, generations + 1):
        print("Generation %d" % g)
        tf.logging.info("generation:%d;cache=%s" % (g, str(analyzer)))
        # initial results from previous data:
        analyzed = list(analyzer.analyze([make_config(x) for x in offsprings]))

        json.dump({"parent": next_parent, "offspring": offsprings},
                  gzip.open(logs_dir + "/run.%05d.json.gz" % g, "wt", encoding="utf8"))

        print(len(next_parent))
        # reduce the number of elements
        filtered_results = next_parent + analyzed
        next_parent = []
        while len(next_parent) < parent_size and len(filtered_results) > 0:
            pareto = PyBspTreeArchive(2).filter([(-x["accuracy"], x["memory"]) for x in filtered_results],
                                                returnIds=True)

            current_pareto = [filtered_results[i] for i in pareto]
            missing = parent_size - len(next_parent)

            if (len(current_pareto) <= missing):
                next_parent += current_pareto
            else:  # distance crowding
                next_parent += crowding_reduce(current_pareto, missing, maximal)

            for i in reversed(sorted(pareto)):
                filtered_results.pop(i)

        parent_conf = [x["quant_conf"] for x in next_parent]

        # generate new candidate solutions:
        offsprings = []
        for i in range(0, offspring_size):
            # select two random parents
            parent = [random.choice(parent_conf), random.choice(parent_conf)]
            child = [8 for _ in range(quantizable_layers)]
            for li in range(quantizable_layers):
                if random.random() < 0.90:  # 90 % probability of crossover
                    child[li] = random.choice(parent)[li]
                else:
                    child[li] = 8

            if random.random() < 0.1:  # 10 % probability of mutation
                li = random.choice([x for x in range(quantizable_layers)])
                child[li] = random.choice([2, 3, 4, 5, 6, 7, 8])

            offsprings.append(child)


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
