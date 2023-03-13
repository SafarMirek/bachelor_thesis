import tensorflow as tf
from keras.applications.resnet import ResNet, stack1
from tensorflow import keras


def ResNet8(
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        **kwargs,
):
    """Instantiates the ResNet50 architecture."""

    def stack_fn(x):
        x = stack1(x, 16, 1, stride1=1, name="conv2")
        x = stack1(x, 16, 1, stride1=1, name="conv3")
        return stack1(x, 32, 1, stride1=1, name="conv4")

    return ResNet(
        stack_fn,
        False,
        True,
        "resnet8",
        True,
        None,
        input_tensor,
        input_shape,
        pooling,
        classes,
        **kwargs,
    )
