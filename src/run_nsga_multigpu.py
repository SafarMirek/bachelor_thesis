from __future__ import print_function

import argparse
import datetime

import keras.models

from nsga.nsga_qat_multigpu import QATNSGA


def main(*, logs_dir, base_model_path, parent_size=25, offspring_size=25, batch_size=64, qat_epochs=6, generations=25,
         previous_run=None, cache_datasets=False):
    base_model = keras.models.load_model(base_model_path)

    nsga = QATNSGA(logs_dir=logs_dir, base_model=base_model, parent_size=parent_size, offspring_size=offspring_size,
                   batch_size=batch_size, qat_epochs=qat_epochs, generations=generations, previous_run=previous_run,
                   cache_datasets=cache_datasets)

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
        default="mobilenet_tinyimagenet.keras",
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
        '--generations',
        type=int,
        default=25,
        help='Number of generations')

    parser.add_argument(
        '--cache-datasets',
        default=False,
        action='store_true',
        help='Cache datasets during QAT')

    args = parser.parse_args()
    main(**vars(args))
