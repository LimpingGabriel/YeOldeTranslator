from Shakespeare import load_shakespeare
from TextDataset import TextDataset
from GlobalSettings import GlobalSettings

from transformer.Transformer import Transformer
from transformer.layers.BaseAttention import *
from transformer.layers.PositionalEmbedding import *
from transformer.layers.CrossAttention import *
from transformer.metrics import *
from transformer.CustomSchedule import *

if __name__ == "__main__":
    settings = GlobalSettings()
    SNShakespeareDataset = TextDataset("SNShakespeare")
    SNShakespeareDataset.prepare()

    t = Transformer(
        num_layers = settings.TRANSFORMER_NUM_LAYERS,
        d_model = settings.TRANSFORMER_D_MODEL,
        num_heads = settings.TRANSFORMER_NUM_HEADS,
        dff = settings.TRANSFORMER_DFF,
        input_vocab_size = SNShakespeareDataset.tokenizers.src.get_vocab_size().numpy(),
        target_vocab_size = SNShakespeareDataset.tokenizers.tar.get_vocab_size().numpy(),
        dropout_rate = settings.TRANSFORMER_DROPOUT_RATE)
    
    train_batches, val_batches = SNShakespeareDataset.make_batches(train_test_split=0.9)
    
    for (src, tar), tar_labels in train_batches.take(1):
        break

    output = t((src, tar))

    learning_rate = CustomSchedule(settings.TRANSFORMER_D_MODEL)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)


    t.summary()



    t.compile(
        loss=masked_loss,
        optimizer=optimizer,
        metrics=[masked_accuracy])


    t.fit(train_batches, epochs=20, validation_data=val_batches)


