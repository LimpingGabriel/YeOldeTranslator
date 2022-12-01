from Shakespeare import *
from TextDataset import TextDataset
from GlobalSettings import GlobalSettings

from transformer.Transformer import Transformer
from transformer.layers.BaseAttention import *
from transformer.layers.PositionalEmbedding import *
from transformer.layers.CrossAttention import *
from transformer.metrics import *

from Translator import *
from ExportTranslator import *

import tensorflow as tf



if __name__ == "__main__":
    settings = GlobalSettings()
    dataset = TextDataset("Shakescleare")
    
    ex = tf.saved_model.load("transformer_batch_size-32__num_layers-6__d_model-128__dff-512__num_heads-8__dropout-0.3__epochs-50_")

    while True:
        print("Output: {}".format(ex(tf.constant(input("Input: ")))))