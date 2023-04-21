# Project: Bachelor Thesis: Automated Quantization of Neural Networks
# Author: Miroslav Safar (xsafar23@fit.vutbr.cz)

from keras.callbacks import Callback


class MaxAccuracyCallback(Callback):
    """
    This callback tracks validation accuracy during training and reports maximal achieved validation accuracy
    """

    def __init__(self):
        super().__init__()
        self.max_accuracy = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_accuracy = logs.get("val_accuracy") or 0
        if self.max_accuracy < val_accuracy:
            self.max_accuracy = val_accuracy

    def get_max_accuracy(self):
        """
        :return: Maximal achieved validation accuracy
        """
        return self.max_accuracy
