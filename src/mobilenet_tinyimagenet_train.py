import argparse
from datetime import datetime

from keras.applications import MobileNet
from tensorflow import keras
import tensorflow as tf

from datasets import tinyimagenet
import os

# Script arguments
parser = argparse.ArgumentParser(
    prog='mobilenet_tinyimagenet_train',
    description='Train Mobilenet for ImageNet subset (100 classes only) datasets',
    epilog='')

parser.add_argument('-e', '--epochs', default=200, type=int)
parser.add_argument('-b', '--batch-size', default=128, type=int)

parser.add_argument('--learning-rate', '--lr', default=0.045, type=float)

parser.add_argument("--logs-dir", default="logs/tinyimagenet/mobilenet/32bit")
parser.add_argument("--checkpoints-dir", default="checkpoints/tinyimagenet/mobilenet/32bit")

parser.add_argument('--from-checkpoint', default=None, type=str)
parser.add_argument('--start-epoch', default=0, type=int)
parser.add_argument('--save-as', default="mobilenet_tinyimagenet.keras", type=str)

parser.add_argument('--cache', default=True, action='store_true')  # on/off flag
parser.add_argument('-v', '--verbose', default=False, action='store_true')  # on/off flag
parser.add_argument('--metal', default=False, action='store_true')


def main(*, epochs, batch_size, learning_rate, logs_dir, checkpoints_dir, from_checkpoint, start_epoch, cache, save_as,
         verbose, metal=False):
    if verbose:
        print("Used configuration:")
        print(f'Start epoch: {start_epoch}')
        print(f'Number of epochs: {epochs}')
        print(f'Batch Size: {batch_size}')
        print(f'Learning rate: {learning_rate}')
        print(f'Checkpoints directory: {checkpoints_dir}')
        print(f'Logs directory: {logs_dir}')
        print(f'Cache training dataset: {cache}')
        if from_checkpoint is not None:
            print(f'From checkpoint: {from_checkpoint}')

    # Create model
    model = MobileNet(input_shape=(224, 224, 3), classes=100, alpha=0.25, weights=None)

    if verbose:
        model.summary()

    # Load dataset
    tr_ds = tinyimagenet.get_tinyimagenet_dataset(split="train")
    tr_ds = tr_ds.map(tinyimagenet.get_preprocess_image_fn(image_size=(224, 224)))

    if cache:
        tr_ds = tr_ds.cache()

    data_augmentation = tf.keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.08),
    ])

    def augment(x):
        if metal:
            # tensorflow-metal has problem with this, using cpu fixes it
            with tf.device("/cpu:0"):
                out = data_augmentation(x, training=True)
                return out
        else:
            return data_augmentation(x, training=True)

    train_ds = tr_ds.map(lambda data: (data['image'], data['label']))

    train_ds = train_ds.shuffle(10000) \
        .batch(batch_size)
    #     .map(lambda x, y: (augment(x), y)) \

    ds = tinyimagenet.get_tinyimagenet_dataset(split="val")
    ds = ds.map(tinyimagenet.get_preprocess_image_fn(image_size=(224, 224)))

    if cache:
        ds = ds.cache()

    test_ds = ds.map(lambda data: (data['image'], data['label'])).batch(batch_size)

    learning_rate_fn = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=2.5 * len(train_ds),
        decay_rate=0.98
    )

    model.compile(optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=learning_rate_fn,
                                                               momentum=0.9),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    if from_checkpoint is not None:
        model.load_weights(from_checkpoint)

    # Define checkpoint callback for saving model weights after each epoch
    checkpoints_dir = os.path.abspath(checkpoints_dir)
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
    logs_dir = os.path.abspath(logs_dir)
    logs_dir = os.path.join(logs_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(logs_dir)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logs_dir)

    model.fit(train_ds, epochs=epochs, validation_data=test_ds, initial_epoch=start_epoch,
              callbacks=[model_checkpoint_callback, tensorboard_callback])

    if save_as is not None:
        model.save(save_as)


if __name__ == "__main__":
    tf.random.set_seed(30082000)  # Set random seed to have reproducible results

    args = parser.parse_args()
    main(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logs_dir=args.logs_dir,
        checkpoints_dir=args.checkpoints_dir,
        from_checkpoint=args.from_checkpoint,
        start_epoch=args.start_epoch,
        cache=args.cache,
        save_as=args.save_as,
        verbose=args.verbose,
        metal=args.metal
    )
