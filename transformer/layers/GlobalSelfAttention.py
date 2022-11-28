from transformer.layers.BaseAttention import BaseAttention

class GlobalSelfAttention(BaseAttention):
    """description of class"""

    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

