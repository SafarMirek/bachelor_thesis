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
from quantization_search.tf_quantization.transforms.quantize_model import quantize_model


# %%

def process_image(data):
    data['image'] = (tf.image.resize(data['image'], (224, 224)) * 2.0 / 255.0) - 1.0
    return data


# %%

model = tf.keras.applications.MobileNet(weights='imagenet', input_shape=(224, 224, 3))

# %%

model.summary()

# %%

model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# %%

tr_ds = tfds.load('imagenet_v2', split='test[:90%]')
tr_ds = tr_ds.map(process_image)

train_ds = tr_ds \
    .map(lambda data: (data['image'], data['label'])) \
    .batch(32)

ds = tfds.load('imagenet_v2', split='test[90%:]')
ds = ds.map(process_image)

test_ds = ds.map(lambda data: (data['image'], data['label'])).batch(64)

# %%

loss, acc = model.evaluate(test_ds)
print(f'Top-1 accuracy (float): {acc * 100:.2f}%')

# %%
print("Quantize model")

# q_aware stands for for quantization aware.
q_aware_model = quantize_model(model, [])

# `quantize_model` requires a recompile.
q_aware_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

q_aware_model.summary()

# %%

q_aware_model.fit(train_ds, epochs=3, validation_data=test_ds)

# %%

qa_loss, qa_acc = q_aware_model.evaluate(test_ds)
print(f'Top-1 accuracy (quantize aware float): {qa_acc * 100:.2f}%')
