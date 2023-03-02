import numpy as np
import tensorflow as tf


def evaluate_tflite_model(*, interpreter: tf.lite.Interpreter, validation_data: tf.data.Dataset):
    """Evaluates TFLite model accuracy"""
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    total = 0
    ok = 0

    for (validation_x, validation_y) in validation_data.as_numpy_iterator():
        for index, x in enumerate(validation_x):
            print("#" + str(total) + " Running...", end="")
            x = np.expand_dims(x, axis=0).astype(np.float32)

            interpreter.set_tensor(input_index, x)

            # Run inference.
            interpreter.invoke()

            output = interpreter.tensor(output_index)
            prediction = np.argmax(output()[0])

            if prediction == validation_y[index]:
                ok = ok + 1
                print("OK")
            else:
                print("FAIL")

            total = total + 1
            if total % 100 == 0:
                print('Evaluated on {n} results so far...'.format(n=total))
    accuracy = ok / total
    return accuracy
