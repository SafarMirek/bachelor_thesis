# Project: Bachelor Thesis: Automated Quantization of Neural Networks
# Author: Miroslav Safar (xsafar23@stud.fit.vutbr.cz)

import argparse

from keras.applications import MobileNet

from resnet import ResNet18


def load_and_save(*, weights_path, destination, classes):
    """
    Loads weights of model from weights file and save model in keras format
    :param weights_path: Path to weights file
    :param destination: Path to save keras model
    :param alpha: alpha of mobilenet
    :param classes: Number of classes that model should distinguish
    """
    model = ResNet18(input_shape=(32, 32, 3), classes=classes)
    model.load_weights(weights_path)
    model.save(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='resnet18_cifar10_create',
        description='Create cifar model from weights file',
        epilog='')

    parser.add_argument("--weights-path", required=True)
    parser.add_argument("--destination", default="resnet18_cifar10.keras", type=str)
    parser.add_argument("--classes", default=10, type=int)

    args = parser.parse_args()

    load_and_save(weights_path=args.weights_path, destination=args.destination, classes=args.classes)
