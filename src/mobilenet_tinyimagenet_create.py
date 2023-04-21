# Project: Bachelor Thesis: Automated Quantization of Neural Networks
# Author: Miroslav Safar (xsafar23@fit.vutbr.cz)

import argparse

from keras.applications import MobileNet


def load_and_save(*, weights_path, destination, alpha, classes):
    """
    Loads weights of model from weights file and save model in keras format
    :param weights_path: Path to weights file
    :param destination: Path to save keras model
    :param alpha: alpha of mobilenet
    :param classes: Number of classes that model should distinguish
    """
    model = MobileNet(input_shape=(224, 224, 3), classes=classes, alpha=alpha, weights=None)
    model.load_weights(weights_path)
    model.save(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='mobilenet_tinyimagenet_create',
        description='Create mobilenet model from weights file',
        epilog='')

    parser.add_argument("--weights-path", required=True)
    parser.add_argument("--destination", default="mobilenet_tinyimagenet.keras", type=str)
    parser.add_argument("--alpha", default=0.25, type=float)
    parser.add_argument("--classes", default=100, type=int)

    args = parser.parse_args()

    load_and_save(weights_path=args.weights_path, destination=args.destination, alpha=args.alpha, classes=args.classes)
