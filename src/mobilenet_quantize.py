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

from tf_quantization.transforms.quantize_model import quantize_model
import imagenet_mini

# %%

model = tf.keras.applications.MobileNet(weights='imagenet', input_shape=(224, 224, 3), alpha=0.25)

# %%

model.summary()

# %%

model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# %%

tr_ds = imagenet_mini.get_imagenet_mini_dataset(split="train")
tr_ds = tr_ds.map(imagenet_mini.get_preprocess_image_fn(image_size=(224, 224)))

train_ds = tr_ds \
    .map(lambda data: (data['image'], data['label'])) \
    .batch(128)

ds = imagenet_mini.get_imagenet_mini_dataset(split="val")
ds = ds.map(imagenet_mini.get_preprocess_image_fn(image_size=(224, 224)))

test_ds = ds.map(lambda data: (data['image'], data['label'])).batch(64)

# %%

loss, acc = model.evaluate(test_ds)
print(f'Top-1 accuracy (float): {acc * 100:.2f}%')

# %%
print("Quantize model")

bit_8_conf = {"weight_bits": 8, "activation_bits": 8}
bit_7_conf = {"weight_bits": 7, "activation_bits": 8}
bit_6_conf = {"weight_bits": 6, "activation_bits": 8}
bit_5_conf = {"weight_bits": 5, "activation_bits": 8}
bit_4_conf = {"weight_bits": 4, "activation_bits": 8}
bit_3_conf = {"weight_bits": 3, "activation_bits": 8}
bit_2_conf = {"weight_bits": 2, "activation_bits": 8}

q_aware_model = quantize_model(model, [
    bit_8_conf, bit_8_conf, bit_8_conf, bit_8_conf, bit_8_conf,
    bit_8_conf, bit_8_conf, bit_8_conf, bit_8_conf, bit_8_conf,
    bit_8_conf, bit_8_conf, bit_8_conf, bit_8_conf, bit_8_conf,
    bit_8_conf, bit_8_conf, bit_8_conf, bit_8_conf, bit_8_conf,
    bit_8_conf, bit_8_conf, bit_8_conf, bit_8_conf, bit_8_conf,
    bit_8_conf, bit_8_conf, bit_8_conf, bit_8_conf, bit_8_conf,
    bit_8_conf, bit_8_conf, bit_8_conf, bit_8_conf, bit_8_conf,
    bit_8_conf, bit_8_conf
])

# `quantize_model` requires a recompile.
q_aware_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.01),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

q_aware_model.summary()

# %%

qa_loss, qa_acc = q_aware_model.evaluate(test_ds)
print(f'Top-1 accuracy before QAT (quantized float): {qa_acc * 100:.2f}%')

# %%

checkpoint_filepath = 'checkpoints/weights-{epoch:03d}-{val_accuracy:.4f}.hdf5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode="max"
)

# Train for 5 epochs
q_aware_model.fit(train_ds, epochs=5, validation_data=test_ds, callbacks=[model_checkpoint_callback], initial_epoch=0)

# %%

qa_loss, qa_acc = q_aware_model.evaluate(test_ds)
print(f'Top-1 accuracy after (quantize aware float): {qa_acc * 100:.2f}%')

# %%
# Save model?
q_aware_model.save("mobilenet_quant.keras")
