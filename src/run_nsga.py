# Project: Bachelor Thesis: Automated Quantization of Neural Networks
# Author: Miroslav Safar (xsafar23@stud.fit.vutbr.cz)

import argparse
import datetime

import keras.models

from nsga.nsga_qat import QATNSGA
from nsga.nsga_qat_multigpu import MultiGPUQATNSGA


def main(*, logs_dir, base_model_path, parent_size=25, offspring_size=25, batch_size=64, qat_epochs=6, generations=25,
         previous_run=None, cache_datasets=False, approx=False, act_quant_wait=0, per_channel=True, symmetric=True,
         learning_rate=0.2, multigpu=False, exhaustive=False, timeloop_architecture="eyeriss", model_name="mobilenet",
         bn_freeze=40):
    """
    Run NSGA-II for searching of best mixed-precision quantization configurations of layers

    :param logs_dir: Directory for logging
    :param base_model_path: Path to base not-quantized model
    :param parent_size: Number of parents
    :param offspring_size: Number of offsprings
    :param batch_size: Batch size
    :param qat_epochs: Number of quantization-aware epochs for partly training of models
    :param generations: Number of generations
    :param previous_run: Continue previous run
    :param cache_datasets: Cache datasets during training
    :param approx: Use Approximate version of QuantFusedConv2DBatchNormalization
    :param act_quant_wait:
    :param per_channel: Weighs are quantized  per-channel
    :param symmetric: Weights are quantized symmetrically
    :param learning_rate: Starting learning rate of quantization aware training
    :param multigpu: Use multiple GPUs
    :param exhaustive: Use exhaustive timeloop heuristic or random heuristic
    """

    timeloop_heuristic = "exhaustive" if exhaustive else "random"

    multigpu = True  # Cache rework is not done for Single GPU Evaluation,

    if multigpu:
        print(f"Initializing QAT NSGA-II MultiGPU for {timeloop_architecture} architecture")
        nsga = MultiGPUQATNSGA(logs_dir=logs_dir, base_model_path=base_model_path, parent_size=parent_size,
                               offspring_size=offspring_size,
                               batch_size=batch_size, qat_epochs=qat_epochs, generations=generations,
                               previous_run=previous_run,
                               cache_datasets=cache_datasets, approx=approx, activation_quant_wait=act_quant_wait,
                               per_channel=per_channel, symmetric=symmetric, learning_rate=learning_rate,
                               timeloop_heuristic=timeloop_heuristic, timeloop_architecture=timeloop_architecture,
                               model_name=model_name, bn_freeze=bn_freeze)
    else:
        print(f"Initializing QAT NSGA-II for {timeloop_architecture} architecture")
        nsga = QATNSGA(logs_dir=logs_dir, base_model_path=base_model_path, parent_size=parent_size,
                       offspring_size=offspring_size,
                       batch_size=batch_size, qat_epochs=qat_epochs, generations=generations,
                       previous_run=previous_run,
                       cache_datasets=cache_datasets, approx=approx, activation_quant_wait=act_quant_wait,
                       per_channel=per_channel, symmetric=symmetric, learning_rate=learning_rate,
                       timeloop_heuristic=timeloop_heuristic, timeloop_architecture=timeloop_architecture,
                       model_name=model_name, bn_freeze=bn_freeze)

    nsga.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--logs-dir',
        type=str,
        default="/tmp/safarmirek/run" + datetime.datetime.now().strftime("-%Y%m%d-%H%M"),
        help='Logs dir')

    parser.add_argument(
        '--previous-run',
        type=str,
        default=None,
        help='Logs dir of previous run to continue')

    parser.add_argument(
        '--base-model-path',
        type=str,
        default="mobilenet_tinyimagenet_025.keras",
        help='')

    parser.add_argument(
        '--architecture',
        type=str,
        default="eyeriss",
        dest="timeloop_architecture",
        help='')

    parser.add_argument(
        '--parent-size',
        type=int,
        default=10,
        help='Number of images in the batch.')

    parser.add_argument(
        '--offspring-size',
        type=int,
        default=10,
        help='Number of images in the batch.')

    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Number of images in the batch.')

    parser.add_argument(
        '--qat-epochs',
        type=int,
        default=10,
        help='Number of QAT epochs on the model for the accuracy eval during NSGA')

    parser.add_argument(
        '--act-quant-wait',
        type=int,
        default=0)

    parser.add_argument('--learning-rate', '--lr', default=0.2, type=float)

    parser.add_argument('--bn-freeze', default=40, type=int)

    parser.add_argument(
        '--generations',
        type=int,
        default=25,
        help='Number of generations')

    parser.add_argument(
        '--cache-datasets',
        default=False,
        action='store_true',
        help='Cache datasets during QAT')

    parser.add_argument('--approx', default=False, action='store_true')
    parser.add_argument('--exhaustive', default=False, action='store_true')
    parser.add_argument('--per-channel', default=False, action='store_true')
    parser.add_argument('--symmetric', default=False, action='store_true')

    parser.add_argument('--multigpu', default=False, action='store_true')

    parser.add_argument(
        '--model-name',
        type=str,
        default="mobilenet",
        dest="model_name",
        help='')

    args = parser.parse_args()

    if args.base_model_path == "resnet18_cifar10.keras" and args.model_name == "mobilenet":
        print("Bad model name changing to resnet.")
        model_name = "resnet"

    main(**vars(args))
