import tensorflow as tf
from transformer.layers.CausalSelfAttention import CausalSelfAttention
from transformer.layers.CrossAttention import CrossAttention
from transformer.layers.FeedForward import FeedForward

class DecoderLayer(tf.keras.layers.Layer):
    """description of class"""

    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim = d_model,
            dropout=dropout_rate)

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim = d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)
        return x
