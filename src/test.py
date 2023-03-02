from tensorflow import keras
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Conv2D, MaxPool2D, Flatten, Dense

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img
from keras.utils import img_to_array
from keras.applications import imagenet_utils
import os

import pandas as pd
import tensorflow_datasets as tfds
import tensorflow_model_optimization as tfmot

from tf_quantization.transforms.quantize_transforms import PerLayerQuantizeModelTransformer

import tiny_imagenet_dataset


def run():
    model = tf.keras.applications.MobileNet(weights='imagenet', input_shape=(32, 32, 3))

    print(model.summary())

    model_transformer = PerLayerQuantizeModelTransformer(model, {})
    layers = model_transformer.get_number_of_quantizable_layers()
    print("Počet quantizovatelných vrstev: " + str(layers))


if __name__ == "__main__":
    run()
