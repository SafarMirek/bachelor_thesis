import tensorflow_datasets as tfds
import tensorflow as tf


def get_imagenet_mini_dataset(split="train"):
    """This method process and returns imagenet-mini dataset"""
    if split == "val":
        return tfds.load('imagenet_v2', split='test', shuffle_files=True)
    elif split == "train":
        return tfds.load('imagenet150', split='train', shuffle_files=True)
    raise ValueError("Split must be train or val")


def preprocess_image(data, image_size=(224, 224)):
    """This method preprocess images to input format of Mobilenet"""
    data['image'] = (tf.image.resize(data['image'], image_size) * 2.0 / 255.0) - 1.0
    return data


def get_preprocess_image_fn(image_size=(224, 224)):
    return lambda data: preprocess_image(data, image_size=image_size)
