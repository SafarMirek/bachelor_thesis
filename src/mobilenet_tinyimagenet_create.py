import argparse

from keras.applications import MobileNet


def load_and_save(weights_path):
    model = MobileNet(input_shape=(224, 224, 3), classes=100, alpha=0.1, weights=None)
    model.load_weights(weights_path)
    model.save("mobilenet_tinyimagenet.keras")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='mobilenet_tinyimagenet_qat',
        description='Quantize mobilenet',
        epilog='')

    parser.add_argument("--weights-path", required=True)

    args = parser.parse_args()
    print(args)

    load_and_save(args.weights_path)
