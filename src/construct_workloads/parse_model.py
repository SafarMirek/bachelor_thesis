# This code has been inspired by the code provided with the Maestro tool: https://github.com/maestro-project/maestro/blob/master/tools/frontend/frameworks_to_modelfile_maestro.py

import os
import argparse
from argparse import RawTextHelpFormatter
import tensorflow as tf
import tensorflow.keras.applications as keras_models
from keras.models import load_model


# Retrieve the layer output shape for models
def get_output_size(W, H, kernel_size, stride, padding):
    W_out = int((W - kernel_size + 2 * padding) / stride) + 1
    H_out = int((H - kernel_size + 2 * padding) / stride) + 1
    # dilation = 1
    # W_out = int((W + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    # H_out = int((H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    return W_out, H_out


# Get layer dimensions from pytorch model summary
def get_conv_layers_keras(model, input_size, batch_size):
    layers = []
    W, H, C = input_size
    N = batch_size

    for m in model.layers:
        if isinstance(m, tf.keras.layers.Conv2D) or isinstance(m, tf.keras.layers.DepthwiseConv2D):
            if isinstance(m, tf.keras.layers.DepthwiseConv2D):
                M = C
            else:
                M = m.filters

            S = m.kernel_size[0]
            R = m.kernel_size[1]
            Wpad = m.padding
            Hpad = m.padding

            if Wpad == 'same':
                Wpad = (S - 1) // 2
                Hpad = (R - 1) // 2
            elif Wpad == 'valid':
                Wpad = 0
                Hpad = 0

            Wstride = m.strides[0]
            Hstride = m.strides[1]

            layer = (W, H, C, N, M, S, R, Wpad, Hpad, Wstride, Hstride)
            layers.append(layer)

            W, H = get_output_size(W, H, S, Wstride, Wpad)
            C = M

        if isinstance(m, tf.keras.layers.MaxPooling2D):
            Wstride = m.strides[0]
            Hstride = m.strides[1]
            W = W // Wstride
            H = H // Hstride
    return layers


# Create Timeloop layer description from parsed keras model
def parse_keras_model(input_size, model_file, batch_size, out_dir, out_file, api_name, verbose=False):
    input_size = tuple((int(d) for d in str.split(input_size, ",")))
    model = load_model(model_file)
    cnn_layers = get_conv_layers_keras(model, input_size, batch_size)

    if verbose:
        print("# Model: " + str(model_file.split(".")[0]))
        print("# W, H, C, N, M, S, R, Wpad, Hpad, Wstride, Hstride")
        print("cnn_layers = [")
        for layer in cnn_layers:
            print("    " + str(layer) + ",")
        print("]")

    with open(os.path.join(out_dir, out_file + ".yaml"), "w") as f:
        f.write(f"api: {api_name}\n")
        f.write(f"model: " + str(model_file.split('.')[0]) + "\n")
        f.write("# W, H, C, N, M, S, R, Wpad, Hpad, Wstride, Hstride\n")
        f.write("layers:\n")
        for layer in cnn_layers:
            f.write("  - [")
            f.write(", ".join(str(p) for p in layer))
            f.write("]\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter, prog="parse_model",
                                     description="Parser of keras/pytorch models into Timeloop layer description format")
    parser.add_argument('-a', '--api_name', type=str, default="keras", help="api choices: pytorch, keras")
    parser.add_argument('-i', '--input_size', type=str, default="224,224,3", help='input size in format W,H,C')
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('-m', '--model_file', type=str, required=True, help='relative path to model file')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('-o', '--outfile', type=str, default=f"parsed_model_layers", help='output file name')

    # Parse arguments
    opt = parser.parse_args()

    if opt.verbose:
        print('Begin processing')
        print('API name: ' + str(opt.api_name))
        print('Model name: ' + str(opt.model))
        print('Input size: ' + str(opt.input_size))

    out_dir = "parsed_models"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Process model based on chosen API and return layer dimensions
    if opt.api_name == 'keras':
        parse_keras_model(opt.input_size, opt.model_file, opt.batch_size, out_dir, opt.outfile, opt.api_name,
                          opt.verbose)
