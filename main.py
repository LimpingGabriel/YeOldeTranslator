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



if __name__ == "__main__":
    settings = GlobalSettings()
    dataset = TextDataset("Shakescleare")
    
    