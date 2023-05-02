# Project: Bachelor Thesis: Automated Quantization of Neural Networks
# Author: Miroslav Safar (xsafar23@stud.fit.vutbr.cz)

import tensorflow_datasets as tfds
import tensorflow as tf


def get_cifar10_dataset(split="train"):
    """This method process and returns cifar10 dataset"""
    return tfds.load('cifar10', split=split, shuffle_files=True)


def preprocess_image(data, image_size=(32, 32)):
    """This method preprocess images to input format of Mobilenet"""
    data['image'] = (tf.image.resize(data['image'], image_size) * 2.0 / 255.0) - 1.0
    return data


def get_preprocess_image_fn(image_size=(32, 32)):
    """
    Returns method for preprocessing image to specified imagesize and values in <-1;1>
    :param image_size: Requested image size
    :return: preprocess_image function
    """
    return lambda data: preprocess_image(data, image_size=image_size)
