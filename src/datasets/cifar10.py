import tensorflow_datasets as tfds
import tensorflow as tf


def get_imagenet_mini_dataset(split="train"):
    """This method process and returns imagenet-mini dataset"""
    return tfds.load('cifar10', split=split, shuffle_files=True)


def preprocess_image(data, image_size=(32, 32)):
    """This method preprocess images to input format of Mobilenet"""
    data['image'] = (tf.image.resize(data['image'], image_size) * 2.0 / 255.0) - 1.0
    return data


def get_preprocess_image_fn(image_size=(32, 32)):
    return lambda data: preprocess_image(data, image_size=image_size)
