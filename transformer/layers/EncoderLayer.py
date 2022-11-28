import tensorflow as tf
from tensorflow.keras.layers import Layer
from transformer.layers.GlobalSelfAttention import GlobalSelfAttention
from transformer.layers.FeedForward import FeedForward

class EncoderLayer(Layer):
    """description of class"""
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads = num_heads,
            key_dim = d_model,
            dropout = dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x



