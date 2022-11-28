import tensorflow as tf
from keras.layers import MultiHeadAttention
from keras.layers import LayerNormalization
from keras.layers import Add

class BaseAttention(tf.keras.layers.Layer):
    """description of class"""
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = MultiHeadAttention(**kwargs)
        self.layernorm = LayerNormalization()
        self.add = Add()




