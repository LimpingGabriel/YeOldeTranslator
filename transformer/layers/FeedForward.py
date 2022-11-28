import tensorflow as tf

from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras import Sequential

class FeedForward(Layer):
    """description of class"""
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = Sequential([
            Dense(dff, activation="relu"),
            Dense(d_model),
            Dropout(dropout_rate)
            ])

        self.add = Add()
        self.layer_norm = LayerNormalization()


    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x

