import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dropout
from transformer.layers.PositionalEmbedding import PositionalEmbedding
from transformer.layers.DecoderLayer import DecoderLayer

class Decoder(Layer):
    """description of class"""

    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size,
            d_model=d_model)
        self.dropout = Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        self.last_attn_scores = None

    def call(self, x, context):
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
        return x




