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
    prog='mobilenet_quantize',
    description='Quantize mobilenet',
    epilog='')

parser.add_argument('-e', '--epochs', default=100, type=int)

parser.add_argument('-b', '--batch-size', default=256, type=int)

parser.add_argument('-v', '--verbose', action='store_true')  # on/off flag


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
            'target_lr': self.target_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'hold': self.hold
        }
        return config


def main():
    args = parser.parse_args()
    print(f'Batch Size: {args.batch_size}')
    model = tf.keras.applications.MobileNet(weights='imagenet', input_shape=(224, 224, 3), alpha=0.25)

    if args.verbose:
        print("Original model")
        model.summary()

    # Load dataset
    tr_ds = imagenet_mini.get_imagenet_mini_dataset(split="train")
    tr_ds = tr_ds.map(imagenet_mini.get_preprocess_image_fn(image_size=(224, 224)))

    train_ds = tr_ds \
        .map(lambda data: (data['image'], data['label'])) \
        .batch(args.batch_size)

    ds = imagenet_mini.get_imagenet_mini_dataset(split="val")
    ds = ds.map(imagenet_mini.get_preprocess_image_fn(image_size=(224, 224)))

    test_ds = ds.map(lambda data: (data['image'], data['label'])).batch(args.batch_size)

    if args.verbose:
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

    q_aware_model.summary()

    total_steps = len(train_ds) * args.epochs
    # 10% of the steps
    warmup_steps = int(0.1 * total_steps)

    schedule = WarmUpCosineDecay(target_lr=0.001, warmup_steps=warmup_steps, total_steps=total_steps, hold=warmup_steps)

    # `quantize_model` requires a recompile.
    q_aware_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=schedule),
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=['accuracy'])

    qa_loss, qa_acc = q_aware_model.evaluate(test_ds)
    print(f'Top-1 accuracy before QAT (quantized float): {qa_acc * 100:.2f}%')

    checkpoints_dir = "checkpoints/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    os.system(f'mkdir -p {checkpoints_dir}')

    checkpoint_filepath = checkpoints_dir + '/weights-{epoch:03d}-{val_accuracy:.4f}.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode="max"
    )

    # Define the Keras TensorBoard callback.
    logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    # Train
    q_aware_model.fit(train_ds, epochs=args.epochs, validation_data=test_ds,
                      callbacks=[model_checkpoint_callback, tensorboard_callback],
                      initial_epoch=0)

    qa_loss, qa_acc = q_aware_model.evaluate(test_ds)
    print(f'Top-1 accuracy after (quantize aware float): {qa_acc * 100:.2f}%')

    # Save model
    q_aware_model.save("mobilenet_quant.keras")


if __name__ == "__main__":
    main()
