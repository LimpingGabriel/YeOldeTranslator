from keras.layers import Layer
from PositionalEmbedding import PositionalEmbedding


class Encoder(Layer):
    """description of class"""

    def __init__(self, *, num_layers, d_model, num_heads, 
                 dff, vocab_size, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(
            vocab_size = vocab_size, d_model = d_model)


