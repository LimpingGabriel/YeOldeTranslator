import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow_text
import tensorflow as tf

MODEL_NAME = "Models/transformer_batch_size-64__num_layers-4__d_model-128__dff-256__num_heads-8__dropout-0.3__epochs-200_"

if __name__ == "__main__":
    ex = tf.saved_model.load(MODEL_NAME)

    while True:
        print("Output: {}".format(ex(tf.constant(input("Input: "))).numpy().decode("utf-8")))