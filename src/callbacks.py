from keras.callbacks import Callback


class MaxAccuracyCallback(Callback):

    def __init__(self):
        super().__init__()
        self.max_accuracy = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_accuracy = logs.get("val_accuracy") or 0
        self.try_new_accuracy(val_accuracy)

    def get_max_accuracy(self):
        return self.max_accuracy

    def try_new_accuracy(self, accuracy):
        if self.max_accuracy < accuracy:
            self.max_accuracy = accuracy
