# Project: Bachelor Thesis: Automated Quantization of Neural Networks
# Author: Miroslav Safar (xsafar23@stud.fit.vutbr.cz)

import argparse
from abc import ABC
from datetime import datetime

from tensorflow import keras
import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper

from callbacks_lib import MaxAccuracyCallback
from tf_quantization.layers.approx.quant_conv2D_batch_layer import ApproxQuantFusedConv2DBatchNormalizationLayer
from tf_quantization.layers.approx.quant_depthwise_conv2d_bn_layer import \
    ApproxQuantFusedDepthwiseConv2DBatchNormalizationLayer
from tf_quantization.layers.quant_conv2D_batch_layer import QuantFusedConv2DBatchNormalizationLayer
from tf_quantization.layers.quant_depthwise_conv2d_bn_layer import QuantFusedDepthwiseConv2DBatchNormalizationLayer
from tf_quantization.quantize_model import quantize_model
from datasets import tinyimagenet
import os
import numpy as np


class WarmUpCosineDecay(keras.optimizers.schedules.LearningRateSchedule, ABC):
    # From https://stackabuse.com/learning-rate-warmup-with-cosine-decay-in-keras-and-tensorflow/
    def __init__(self, target_lr, warmup_steps, total_steps, hold):
        super().__init__()
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.hold = hold

    def __call__(self, step):
        lr = self._lr_warmup_cosine_decay(global_step=step,
                                          total_steps=self.total_steps,
                                          warmup_steps=self.warmup_steps,
                                          target_lr=self.target_lr,
                                          hold=self.hold)

        return tf.where(
            step > self.total_steps, 0.0, lr, name="learning_rate"
        )

    @staticmethod
    def _lr_warmup_cosine_decay(global_step,
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

    def get_config(self):
        config = {
            'target_lr': self.target_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'hold': self.hold
        }
        return config


def main(*, q_aware_model, epochs, eval_epochs, bn_freeze=10e1000, batch_size=128, learning_rate=0.05, warmup=0.0,
         checkpoints_dir=None, logs_dir=None,
         cache_dataset=True, from_checkpoint=None, verbose=False, start_epoch=0, activation_quant_wait=0,
         save_best_only=False):
    """
    Run quantization-aware training with provided quantized model for tinyimagenet dataset and returns the best accuracy
    achieved during training
    :param q_aware_model: Provided model with fake quantization
    :param epochs: Number of epochs for the training
    :param eval_epochs: Number of maximal epochs for calculating cos decay of learning rate
    :param bn_freeze: Epoch when should the model's batch normalizations be freezed
    :param batch_size: Size of the batch
    :param learning_rate: Starting learning rate
    :param warmup: Warmup percent
    :param checkpoints_dir: Directory for checkpoints
    :param logs_dir: Directory for training logs
    :param cache_dataset: If the dataset should be cached during training
    :param from_checkpoint: Continue from checkpoint
    :param verbose: Should be verbose
    :param start_epoch: Continue from epoch
    :param activation_quant_wait: When activation quantization should be activated
    :param save_best_only: Save only best checkpoints
    :return: Best achieved accuracy during QA training
    """
    if verbose:
        print("Used configuration:")
        print(f'Number of epochs: {epochs}')
        print(f'Batch Size: {batch_size}')
        print(f'Learning rate: {learning_rate}')
        print(f'Warmup: {warmup}')
        print(f'Checkpoints directory: {checkpoints_dir}')
        print(f'Logs directory: {logs_dir}')
        print(f'Cache training dataset: {cache_dataset}')
        if from_checkpoint is not None:
            print(f'From checkpoint: {from_checkpoint}')

    if warmup < 0 or warmup > 1:
        raise ValueError("Warmup % must be in <0,1> interval.")

    if bn_freeze < activation_quant_wait:
        raise ValueError("BN Freeze before activation quant is not supported")

    # Load tinyimagenet training and validation dataset
    tr_ds = tinyimagenet.get_tinyimagenet_dataset(split="train")
    tr_ds = tr_ds.map(tinyimagenet.get_preprocess_image_fn(image_size=(224, 224)))

    if cache_dataset:
        tr_ds = tr_ds.cache()

    train_ds = tr_ds.map(lambda data: (data['image'], data['label']))
    train_ds = train_ds.shuffle(10000, seed=170619).batch(batch_size).prefetch(32)

    ds = tinyimagenet.get_tinyimagenet_dataset(split="val")
    ds = ds.map(tinyimagenet.get_preprocess_image_fn(image_size=(224, 224)))

    if cache_dataset:
        ds = ds.cache()

    test_ds = ds.map(lambda data: (data['image'], data['label'])).batch(batch_size).prefetch(32)

    if from_checkpoint is not None:
        q_aware_model.load_weights(from_checkpoint)

    # Use Cosine Decay for learning rate
    # Total steps are calculated from eval epochs and not from epochs itself
    # That allows us to partially train the model with same learning rate as the model
    # will be trained in the final evaluation
    total_steps = len(train_ds) * eval_epochs
    warmup_steps = int(warmup * total_steps)

    schedule = WarmUpCosineDecay(target_lr=learning_rate, warmup_steps=warmup_steps, total_steps=total_steps,
                                 hold=warmup_steps)

    if not from_checkpoint and activation_quant_wait == 0:
        # Train activation moving averages
        # When activation quantization is activated since the start of the training
        # we need to train the model with 0 learning rate for few epochs to achieve
        # better results (quantization ranges are starting from <-6,6> and needs to be adjusted for the network)

        q_aware_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0),
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=['accuracy'])

        q_aware_model.fit(train_ds, epochs=3)

    q_aware_model.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=schedule, momentum=0.9),
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=['accuracy'])

    if activation_quant_wait > 0:
        # Disable activation quantization for the first epochs
        _set_act_quant_no_affect(q_aware_model, True)

    # Define checkpoint callback for saving model weights after each epoch (resp. best weights only)
    callbacks = []
    max_accuracy_callback = MaxAccuracyCallback()
    callbacks.append(max_accuracy_callback)

    if checkpoints_dir is not None:
        checkpoints_dir = os.path.abspath(checkpoints_dir)
        checkpoints_dir = os.path.join(checkpoints_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(checkpoints_dir)

        checkpoint_filepath = checkpoints_dir + '/weights-{epoch:03d}-{val_accuracy:.4f}.hdf5'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode="max",
            save_best_only=save_best_only
        )
        callbacks.append(model_checkpoint_callback)

    # Define the Keras TensorBoard callback
    if logs_dir is not None:
        logs_dir = os.path.abspath(logs_dir)
        logs_dir = os.path.join(logs_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(logs_dir)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logs_dir)
        callbacks.append(tensorboard_callback)

    dis_act_quant_epochs = min(epochs, activation_quant_wait)
    not_frozen_epochs = min(epochs, bn_freeze)

    # Train with not frozen batch norms
    q_aware_model.fit(train_ds, epochs=dis_act_quant_epochs, validation_data=test_ds,
                      callbacks=callbacks,
                      initial_epoch=start_epoch)

    if epochs > activation_quant_wait:
        # Train with quantization of activations enabled and with not frozen batch norms
        _set_act_quant_no_affect(q_aware_model, False)
        q_aware_model.fit(train_ds, epochs=not_frozen_epochs, validation_data=test_ds,
                          callbacks=callbacks,
                          initial_epoch=max(dis_act_quant_epochs, start_epoch))

    if epochs > bn_freeze:
        # Train with bn frozen and with quantization of activations enabled
        _freeze_bn_in_model(q_aware_model)
        q_aware_model.fit(train_ds, epochs=epochs, validation_data=test_ds,
                          callbacks=callbacks,
                          initial_epoch=max(bn_freeze, start_epoch))

    if verbose:
        qa_loss, qa_acc = q_aware_model.evaluate(test_ds)
        print(f'Top-1 accuracy after (quantized float): {qa_acc * 100:.2f}%')
        print(f'Max accuracy during training was: {max_accuracy_callback.get_max_accuracy() * 100:.2f}%')

    return max_accuracy_callback.get_max_accuracy()


def _freeze_bn_in_model(model: keras.models.Model):
    """
    Freezes batch normalization in the network
    :param model: Model with fake quantization and batch normalization
    """
    for layer in model.layers:
        if (isinstance(layer, QuantFusedConv2DBatchNormalizationLayer) or
                isinstance(layer, ApproxQuantFusedConv2DBatchNormalizationLayer) or
                isinstance(layer, QuantFusedDepthwiseConv2DBatchNormalizationLayer) or
                isinstance(layer, ApproxQuantFusedDepthwiseConv2DBatchNormalizationLayer)):
            layer.freeze_bn()
        if isinstance(layer, QuantizeWrapper):
            if isinstance(layer.layer, keras.layers.BatchNormalization):
                layer.trainable = False
                layer.training = False


def _set_act_quant_no_affect(model, value):
    """
    Enables/Disables quantization of activations in the network
    :param model: Model with fake quantization
    :param value: True to disable act quantization and False to enable it
    """
    for layer in model.layers:
        if isinstance(layer, QuantizeWrapper):
            if hasattr(layer.quantize_config, "activation_quantizer"):
                if hasattr(layer.quantize_config.activation_quantizer, "no_affect"):
                    layer.quantize_config.activation_quantizer.no_affect = value


if __name__ == "__main__":
    tf.random.set_seed(30082000)  # Set random seed to have reproducible results

    # Script arguments
    parser = argparse.ArgumentParser(
        prog='mobilenet_tinyimagenet_qat',
        description='Quantize mobilenet',
        epilog='')

    parser.add_argument('-e', '--epochs', default=50, type=int)
    parser.add_argument('--bn-freeze', default=25, type=int)
    parser.add_argument('--act-quant-wait', default=3, type=int)
    parser.add_argument('-b', '--batch-size', default=128, type=int)

    parser.add_argument('--weight-bits', '--wb', default=8, type=int)

    parser.add_argument('--learning-rate', '--lr', default=0.01, type=float)
    parser.add_argument('--warmup', default=0.05, type=float)

    parser.add_argument("--logs-dir", default="logs/tinyimagenet/mobilenet/8bit")
    parser.add_argument("--checkpoints-dir", default="checkpoints/tinyimagenet/mobilenet/8bit")

    parser.add_argument('--from-checkpoint', default=None, type=str)
    parser.add_argument('--start-epoch', default=0, type=int)

    parser.add_argument('-v', '--verbose', default=False, action='store_true')  # on/off flag
    parser.add_argument('--cache', default=False, action='store_true')  # on/off flag
    parser.add_argument('--mobilenet-path', default="mobilenet_tinyimagenet_025.keras", type=str)
    parser.add_argument('--approx', default=False, action='store_true')
    parser.add_argument('--per-channel', default=False, action='store_true')
    parser.add_argument('--symmetric', default=False, action='store_true')

    args = parser.parse_args()
    loaded_model = keras.models.load_model(args.mobilenet_path)

    quant_layer_conf = {"weight_bits": args.weight_bits, "activation_bits": 8}
    qa_model = quantize_model(loaded_model, [quant_layer_conf for _ in range(37)], approx=args.approx,
                              per_channel=args.per_channel, symmetric=args.symmetric)

    main(
        q_aware_model=qa_model,
        epochs=args.epochs,
        eval_epochs=args.epochs,
        bn_freeze=args.bn_freeze,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup=args.warmup,
        checkpoints_dir=args.checkpoints_dir,
        logs_dir=args.logs_dir,
        cache_dataset=args.cache,
        from_checkpoint=args.from_checkpoint,
        verbose=args.verbose,
        start_epoch=args.start_epoch,
        activation_quant_wait=args.act_quant_wait,
    )
