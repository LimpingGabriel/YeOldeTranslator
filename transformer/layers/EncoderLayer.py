from keras.layers import Layer
from GlobalSelfAttention import GlobalSelfAttention
from FeedForward import FeedForward

class EncoderLayer(Layer):
    """description of class"""
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads = num_heads,
            key_dim = d_model,
            dropout_rate = dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x



