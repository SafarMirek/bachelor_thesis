# This script takes nsga output and evalutes the results with longed quantization aware training to achieve
# more accurate results
import argparse
import gzip
import json
import os

import tensorflow as tf
from paretoarchive.core import PyBspTreeArchive
from tensorflow import keras

from nsga.nsga_qat import QATAnalyzer
from nsga.nsga_qat_multigpu import MultiGPUQATAnalyzer


def main(output_file, run, batch_size, qat_epochs, bn_freeze, activation_quant_wait, learning_rate, warmup,
         mobilenet_path, multigpu, approx, per_channel, symmetric, checkpoints_dir_pattern, logs_dir_pattern):
    if multigpu:
        analyzer = MultiGPUQATAnalyzer(batch_size=batch_size, qat_epochs=qat_epochs, bn_freeze=bn_freeze,
                                       learning_rate=learning_rate, warmup=warmup,
                                       activation_quant_wait=activation_quant_wait,
                                       approx=approx, per_channel=per_channel, symmetric=symmetric,
                                       logs_dir_pattern=logs_dir_pattern,
                                       checkpoints_dir_pattern=checkpoints_dir_pattern)
    else:
        base_model = keras.models.load_model(mobilenet_path)
        analyzer = QATAnalyzer(base_model, batch_size=batch_size, qat_epochs=qat_epochs, bn_freeze=bn_freeze,
                               learning_rate=learning_rate, warmup=warmup,
                               activation_quant_wait=activation_quant_wait,
                               approx=approx, per_channel=per_channel, symmetric=symmetric,
                               logs_dir_pattern=logs_dir_pattern,
                               checkpoints_dir_pattern=checkpoints_dir_pattern)

    if output_file is None:
        output_file = os.path.join(os.path.dirname(run), "eval." + str(os.path.basename(run)))

    assert not os.path.isfile(output_file)

    print("# loading %s" % run)
    pr = json.load(gzip.open(run))
    next_parent = pr["parent"]
    offsprings = pr["offspring"]
    merged = next_parent + offsprings

    pareto_ids = PyBspTreeArchive(2).filter([(-x["accuracy"], x["memory"]) for x in merged], returnIds=True)
    pareto = [merged[i] for i in pareto_ids]

    # analyze parents
    analyzed_pareto = list(analyzer.analyze(pareto))

    json.dump(analyzed_pareto, gzip.open(output_file, "wt", encoding="utf8"))


if __name__ == "__main__":
    tf.random.set_seed(30082000)  # Set random seed to have reproducible results

    # Script arguments
    parser = argparse.ArgumentParser(
        prog='nsga_evaluate',
        description='Evaluate results of nsga for qat',
        epilog='')

    parser.add_argument("--output-file", "-o", default=None)
    parser.add_argument("--run", required=True)

    parser.add_argument('-e', '--epochs', default=100, type=int)
    parser.add_argument('-b', '--batch-size', default=256, type=int)
    parser.add_argument('--bn-freeze', default=60, type=int)
    parser.add_argument('--act-quant-wait', default=40, type=int)

    parser.add_argument('--learning-rate', '--lr', default=0.2, type=float)
    parser.add_argument('--warmup', default=0.05, type=float)

    parser.add_argument('--multigpu', default=False, action='store_true')
    parser.add_argument('--mobilenet-path', default="mobilenet_tinyimagenet.keras", type=str)

    parser.add_argument('--approx', default=False, action='store_true')
    parser.add_argument('--per-channel', default=False, action='store_true')
    parser.add_argument('--symmetric', default=False, action='store_true')

    parser.add_argument("--logs-dir-pattern", default="logs/mobilenet/%s")
    parser.add_argument("--checkpoints-dir-pattern", default="checkpoints/mobilenet/%s")

    args = parser.parse_args()

    main(output_file=args.output_file,
         run=args.run, batch_size=args.batch_size, qat_epochs=args.epochs,
         bn_freeze=args.bn_freeze, activation_quant_wait=args.act_quant_wait, learning_rate=args.learning_rate,
         warmup=args.warmup, mobilenet_path=args.mobilenet_path, multigpu=args.multigpu, approx=args.approx,
         per_channel=args.per_channel, symmetric=args.symmetric, logs_dir_pattern=args.logs_dir_pattern,
         checkpoints_dir_pattern=args.checkpoints_dir_pattern)
