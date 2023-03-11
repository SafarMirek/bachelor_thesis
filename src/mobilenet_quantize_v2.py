import argparse
from abc import ABC
from datetime import datetime

from tensorflow import keras
import tensorflow as tf

from tf_quantization.quantize_model import quantize_model
import imagenet_mini
import os
import numpy as np

tf.random.set_seed(30082000)  # Set random seed to have reproducible results

# Script arguments
parser = argparse.ArgumentParser(
    prog='mobilenet_quantize',
    description='Quantize mobilenet',
    epilog='')

parser.add_argument('-e', '--epochs', default=50, type=int)
parser.add_argument('-b', '--batch-size', default=256, type=int)

parser.add_argument('--weight-bits', '--wb', default=8, type=int)

parser.add_argument('--learning-rate', '--lr', default=0.001, type=float)
parser.add_argument('--warmup', default=0.0, type=float)

parser.add_argument("--logs-dir", default="logs/fit/")
parser.add_argument("--checkpoints-dir", default="checkpoints/")

parser.add_argument('--from-checkpoint', default=None, type=str)
parser.add_argument('--start-epoch', default=0, type=int)

parser.add_argument('-v', '--verbose', default=False, action='store_true')  # on/off flag
parser.add_argument('--cache', default=False, action='store_true')  # on/off flag


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
    if args.verbose:
        print("Used configuration:")
        print(f'Start epoch: {args.start_epoch}')
        print(f'Number of epochs: {args.epochs}')
        print(f'Batch Size: {args.batch_size}')
        print(f'Weights bits: {args.weight_bits}')
        print(f'Learning rate: {args.learning_rate}')
        print(f'Warmup: {args.warmup}')
        print(f'Checkpoints directory: {args.checkpoints_dir}')
        print(f'Logs directory: {args.logs_dir}')
        print(f'Cache training dataset: {args.cache}')
        if args.from_checkpoint is not None:
            print(f'From checkpoint: {args.from_checkpoint}')

    if args.weight_bits < 2 or args.weight_bits > 8:
        raise ValueError("Weight bits must be in <2,8> interval.")

    if args.warmup < 0 or args.warmup > 1:
        raise ValueError("Warmup % must be in <0,1> interval.")

    model = tf.keras.applications.MobileNet(weights='imagenet', input_shape=(224, 224, 3), alpha=0.25)

    if args.verbose:
        print("Original model")
        model.summary()

    # Load dataset
    tr_ds = imagenet_mini.get_imagenet_mini_dataset(split="train")
    tr_ds = tr_ds.map(imagenet_mini.get_preprocess_image_fn(image_size=(224, 224)))

    if args.cache:
        tr_ds = tr_ds.cache()

    train_ds = tr_ds.map(lambda data: (data['image'], data['label']))

    train_ds = train_ds.shuffle(10000).batch(args.batch_size)

    ds = imagenet_mini.get_imagenet_mini_dataset(split="val")
    ds = ds.map(imagenet_mini.get_preprocess_image_fn(image_size=(224, 224)))

    if args.cache:
        ds = ds.cache()

    test_ds = ds.map(lambda data: (data['image'], data['label'])).batch(args.batch_size)

    if args.verbose:
        print("Quantize model")

    quant_layer_conf = {"weight_bits": args.weight_bits, "activation_bits": 8}

    q_aware_model = quantize_model(model, [quant_layer_conf for i in range(37)])

    q_aware_model.summary()

    if args.from_checkpoint is not None:
        q_aware_model.load_weights(args.from_checkpoint)

    total_steps = len(train_ds) * args.epochs
    # 10% of the steps
    warmup_steps = int(args.warmup * total_steps)  # do not use warmup, only cosine decay

    schedule = WarmUpCosineDecay(target_lr=args.learning_rate, warmup_steps=warmup_steps, total_steps=total_steps,
                                 hold=warmup_steps)

    q_aware_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0),
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=['accuracy'])

    if not args.from_checkpoint:
        # Train activation moving averages
        q_aware_model.fit(train_ds, epochs=2)
        q_aware_model.save("q_aware_model_after_fit0.h5")

    qa_loss, qa_acc = q_aware_model.evaluate(test_ds)
    print(f'Top-1 accuracy before QAT (quantized float): {qa_acc * 100:.2f}%')

    q_aware_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=schedule),
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=['accuracy'])

    # Define checkpoint callback for saving model weights after each epoch
    checkpoints_dir = os.path.abspath(args.checkpoints_dir)
    checkpoints_dir = os.path.join(checkpoints_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(checkpoints_dir)

    checkpoint_filepath = checkpoints_dir + '/weights-{epoch:03d}-{val_accuracy:.4f}.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode="max"
    )

    # Define the Keras TensorBoard callback.
    logs_dir = os.path.abspath(args.logs_dir)
    logs_dir = os.path.join(logs_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(logs_dir)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logs_dir)

    # Train
    q_aware_model.fit(train_ds, epochs=args.epochs, validation_data=test_ds,
                      callbacks=[model_checkpoint_callback, tensorboard_callback],
                      initial_epoch=args.start_epoch)

    qa_loss, qa_acc = q_aware_model.evaluate(test_ds)
    print(f'Top-1 accuracy after (quantize aware float): {qa_acc * 100:.2f}%')


if __name__ == "__main__":
    main()
