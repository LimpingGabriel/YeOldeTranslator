from Shakespeare import *
from TextDataset import TextDataset
from GlobalSettings import GlobalSettings

from transformer.Transformer import Transformer
from transformer.layers.BaseAttention import *
from transformer.layers.PositionalEmbedding import *
from transformer.layers.CrossAttention import *
from transformer.metrics import *
from transformer.CustomSchedule import *
from Translator import *
from ExportTranslator import *

if __name__ == "__main__":
    settings = GlobalSettings()
    SNShakespeareDataset = TextDataset("Shakescleare")
    SNShakespeareDataset.prepare()

    t = Transformer(
        num_layers = settings.TRANSFORMER_NUM_LAYERS,
        d_model = settings.TRANSFORMER_D_MODEL,
        num_heads = settings.TRANSFORMER_NUM_HEADS,
        dff = settings.TRANSFORMER_DFF,
        input_vocab_size = SNShakespeareDataset.tokenizers.src.get_vocab_size().numpy(),
        target_vocab_size = SNShakespeareDataset.tokenizers.tar.get_vocab_size().numpy(),
        dropout_rate = settings.TRANSFORMER_DROPOUT_RATE)
    
    train_batches, val_batches = SNShakespeareDataset.make_batches(train_test_split=0.95)
    
    for (src, tar), tar_labels in train_batches.take(1):
        break

    output = t((src, tar))

    learning_rate = CustomSchedule(settings.TRANSFORMER_D_MODEL)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)


    t.summary()

    BATCH_SIZE = settings.BATCH_SIZE
    STEPS_PER_EPOCH = train_batches.cardinality().numpy()
    SAVE_PERIOD = 5

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="checkpoints/", 
        verbose=1, 
        save_weights_only=True,
        save_freq= int(SAVE_PERIOD * STEPS_PER_EPOCH))


    t.compile(
        loss=masked_loss,
        optimizer=optimizer,
        metrics=[masked_accuracy])


    t.fit(
        train_batches, 
        epochs=settings.NUM_EPOCHS, 
        validation_data=val_batches,
        callbacks=[cp_callback])

    translator = Translator(SNShakespeareDataset.tokenizers, t)

    ex = ExportTranslator(translator)

    tf.saved_model.save(ex, export_dir = "transformer_export")

    while True:
        sentence = input("Sentence: ")
        translated_text = ex(tf.constant(sentence))
        print(translated_text.numpy())
