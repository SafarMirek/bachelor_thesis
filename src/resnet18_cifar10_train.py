import argparse
from datetime import datetime

from keras.layers.preprocessing.image_preprocessing import HORIZONTAL_AND_VERTICAL, HORIZONTAL
from tensorflow import keras
import tensorflow as tf

from datasets import cifar10
import os

from resnet import ResNet18


def main(*, epochs, batch_size, learning_rate, logs_dir, checkpoints_dir, from_checkpoint, start_epoch, cache, save_as,
         verbose):
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
    model = ResNet18(input_shape=(32, 32, 3), classes=10, weight_decay=1e-4)

    if verbose:
        model.summary()

    # Load dataset
    tr_ds = cifar10.get_cifar10_dataset(split="train")
    tr_ds = tr_ds.map(cifar10.get_preprocess_image_fn(image_size=(32, 32)))

    if cache:
        tr_ds = tr_ds.cache()

    data_augmentation = tf.keras.Sequential([
        keras.layers.RandomFlip(HORIZONTAL, seed=30082000),
        keras.layers.RandomTranslation(height_factor=4 / 32, width_factor=4 / 32, seed=30082000),
        keras.layers.RandomRotation(0.1, seed=30082000),
    ])

    def augment(x):
        with tf.device("/cpu:0"):  # TODO: Remove for final version, this fixes problem with tensorflow-metal
            out = data_augmentation(x, training=True)
            return out

    train_ds = tr_ds.map(lambda data: (data['image'], data['label']))
    train_ds = train_ds.shuffle(int(0.5 * len(train_ds)), reshuffle_each_iteration=True, seed=30082000) \
        .map(lambda x, y: (augment(x), y)) \
        .batch(batch_size) \
        .prefetch(100)

    ds = cifar10.get_cifar10_dataset(split="test")
    ds = ds.map(cifar10.get_preprocess_image_fn(image_size=(32, 32)))

    if cache:
        ds = ds.cache()

    test_ds = ds.map(lambda data: (data['image'], data['label'])).batch(batch_size)

    boundaries = [50 * len(train_ds), 100 * len(train_ds), 150 * len(train_ds)]
    values = [learning_rate * x for x in [1, 0.1, 0.01, 0.001]]
    learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values)

    model.compile(
        optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate_fn, momentum=0.9, nesterov=False),
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


def restore_model_from_checkpoint(from_checkpoint):
    model = ResNet18(input_shape=(32, 32, 3), classes=10)
    model.load_weights(from_checkpoint)
    return model


if __name__ == "__main__":
    tf.random.set_seed(30082000)  # Set random seed to have reproducible results

    # Script arguments
    parser = argparse.ArgumentParser(
        prog='resnet18_cifar10_train',
        description='Train resnet18 for cifar10 datasets',
        epilog='')

    parser.add_argument('-e', '--epochs', default=200, type=int)
    parser.add_argument('-b', '--batch-size', default=64, type=int)

    parser.add_argument('--learning-rate', '--lr', default=0.1, type=float)

    parser.add_argument("--logs-dir", default="logs/cifar10/resnet18/32bit")
    parser.add_argument("--checkpoints-dir", default="checkpoints/cifar10/resnet18/32bit")

    parser.add_argument('--from-checkpoint', default=None, type=str)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--save-as', default="resnet18.keras", type=str)

    parser.add_argument('--cache', default=True, action='store_true')  # on/off flag
    parser.add_argument('-v', '--verbose', default=False, action='store_true')  # on/off flag

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
        verbose=args.verbose
    )
