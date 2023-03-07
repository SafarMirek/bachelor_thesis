import argparse
from abc import ABC
from datetime import datetime

from tensorflow import keras
import tensorflow as tf

from tf_quantization.quantize_model import quantize_model
import imagenet_mini
import os
import numpy as np

parser = argparse.ArgumentParser(
    prog='mobilenet_model_evaluate',
    description='Quantize mobilenet evaluation',
    epilog='')

parser.add_argument('-b', '--batch-size', default=256, type=int)

parser.add_argument('--wb', '--weight_bits', default=8, type=int)

parser.add_argument('--from-checkpoint', default=None, type=str)


def lr_warmup_cosine_decay(global_step,
                           warmup_steps,
                           hold=0,
                           total_steps=0,
                           target_lr=1e-3):
    # From https://stackabuse.com/learning-rate-warmup-with-cosine-decay-in-keras-and-tensorflow/
    # Cosine decay
    # There is no tf.pi so we wrap np.pi as a TF constant
    global_step = tf.dtypes.cast(global_step, dtype=tf.float32)
    learning_rate = 0.5 * target_lr * (1 + tf.cos(
        tf.constant(np.pi) * (global_step - warmup_steps - hold) / float(
            total_steps - warmup_steps - hold)))

    # Target LR * progress of warmup (=1 at the final warmup step)
    warmup_lr = target_lr * (global_step / warmup_steps)

    # Choose between `warmup_lr`, `target_lr` and `learning_rate` based on whether `global_step < warmup_steps` and we're still holding.
    # i.e. warm up if we're still warming up and use cosine decayed lr otherwise
    if hold > 0:
        learning_rate = tf.where(global_step > warmup_steps + hold,
                                 learning_rate, target_lr)

    learning_rate = tf.where(global_step < warmup_steps, warmup_lr, learning_rate)
    return learning_rate


class WarmUpCosineDecay(keras.optimizers.schedules.LearningRateSchedule, ABC):
    # From https://stackabuse.com/learning-rate-warmup-with-cosine-decay-in-keras-and-tensorflow/
    def __init__(self, target_lr, warmup_steps, total_steps, hold):
        super().__init__()
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.hold = hold

    def __call__(self, step):
        lr = lr_warmup_cosine_decay(global_step=step,
                                    total_steps=self.total_steps,
                                    warmup_steps=self.warmup_steps,
                                    target_lr=self.target_lr,
                                    hold=self.hold)

        return tf.where(
            step > self.total_steps, 0.0, lr, name="learning_rate"
        )

    def get_config(self):
        config = {
            'start_lr': self.start_lr,
            'target_lr': self.target_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'hold': self.hold
        }
        return config


def main():
    args = parser.parse_args()
    print(f'Batch Size: {args.batch_size}, Weights bits: {args.weight_bits}')
    model = tf.keras.applications.MobileNet(weights='imagenet', input_shape=(224, 224, 3), alpha=0.25)

    # Load testing dataset
    ds = imagenet_mini.get_imagenet_mini_dataset(split="val")
    ds = ds.map(imagenet_mini.get_preprocess_image_fn(image_size=(224, 224)))

    test_ds = ds.map(lambda data: (data['image'], data['label'])).batch(args.batch_size)

    quant_layer_conf = {"weight_bits": args.weight_bits, "activation_bits": 8}

    q_aware_model = quantize_model(model, [quant_layer_conf for i in range(37)])

    q_aware_model.summary()

    q_aware_model.load_weights(args.from_checkpoint)

    # `quantize_model` requires a recompile.
    q_aware_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.00000001),
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=['accuracy'])

    qa_loss, qa_acc = q_aware_model.evaluate(test_ds)
    print(f'Top-1 accuracy (quantized float): {qa_acc * 100:.2f}%')


if __name__ == "__main__":
    main()
