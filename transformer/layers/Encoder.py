import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dropout
from transformer.layers.PositionalEmbedding import PositionalEmbedding
from transformer.layers.EncoderLayer import EncoderLayer

class Encoder(Layer):
    """description of class"""

    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(
            vocab_size = vocab_size,
            d_model = d_model)

        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        self.dropout = Dropout(dropout_rate)


    def call(self, x):
        x = self.pos_embedding(x)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x




