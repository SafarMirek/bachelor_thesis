# Project: Bachelor Thesis: Automated Quantization of Neural Networks
# Author: Miroslav Safar (xsafar23@stud.fit.vutbr.cz)

import tensorflow_datasets as tfds
import tensorflow as tf


def get_tinyimagenet_dataset(split="train"):
    """This method process and returns imagenet-mini dataset"""
    if split == "val":
        return tfds.load('tiny_imagenet100', split='validation', shuffle_files=True)
    elif split == "train":
        return tfds.load('tiny_imagenet100', split='train', shuffle_files=True)
    raise ValueError("Split must be train or val")


def preprocess_image(data, image_size=(224, 224)):
    """This method preprocess images to input format of Mobilenet"""
    data['image'] = (tf.image.resize(data['image'], image_size) * 2.0 / 255.0) - 1.0
    return data


def get_preprocess_image_fn(image_size=(224, 224)):
    """
    Returns method for preprocessing image to specified imagesize and values in <-1;1>
    :param image_size: Requested image size
    :return: preprocess_image function
    """
    return lambda data: preprocess_image(data, image_size=image_size)
