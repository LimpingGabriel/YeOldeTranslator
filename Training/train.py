import tensorflow as tf
import tensorflow_datasets as tfds

from nltk.translate.bleu_score import corpus_bleu
from TextDataset import TextDataset

from transformer.Transformer import *
from transformer.CustomSchedule import *
from transformer.metrics import *
from Translator import *
from ExportTranslator import *

def train_model(dataset, parameters):
    """Must prepare the split for the dataset first."""
    train_batches, val_batches = dataset.make_batches(parameters["batch_size"])

    t = Transformer(
        num_layers = parameters["num_layers"],
        d_model = parameters["d_model"],
        num_heads = parameters["num_heads"],
        dff = parameters["dff"],
        input_vocab_size = dataset.tokenizers.src.get_vocab_size().numpy(),
        target_vocab_size = dataset.tokenizers.tar.get_vocab_size().numpy(),
        dropout_rate = parameters["dropout"])

    # Call the model to get the output size and network shape
    for (src, tar), tar_labels in train_batches.take(1):
        break
    output = t((src, tar))
    settings.logger.debug("Loaded sample data into model.")

    learning_rate = CustomSchedule(parameters["d_model"])
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
    settings.logger.debug("Created custom learning rate.")

    t.summary()

    t.compile(
        loss=masked_loss,
        optimizer=optimizer,
        metrics=[masked_accuracy])
    settings.logger.debug("Compiled model.")


    try:
        
        hist = t.fit(
            train_batches, 
            epochs=parameters["epochs"], 
            validation_data=val_batches)
        translator = Translator(dataset.tokenizers, t)

        ex = ExportTranslator(translator)

        tf.saved_model.save(ex, 
            export_dir="transformer"+ ''.join(["_{}-{}_".format(key, value) for key, value in parameters.items()]))
        settings.logger.debug("Saved model.")
        

        val_loss = hist.history["val_loss"][-1]
        val_acc = hist.history["val_masked_accuracy"][-1]
        
        settings.logger.debug("Calculated val_loss as: {}".format(val_loss))
        settings.logger.debug("Calculated val_acc as: {}".format(val_acc))
        

        reference_corpus = []
        candidate_corpus = []

        sentences = dataset.test_raw.batch(1).take(dataset.test_raw.cardinality())
        settings.logger.debug("Batched {} sentences for BLEU scoring.".format(dataset.test_raw.cardinality()))

        
        ref_tensors = tf.concat([reference for _, reference in sentences], 0)
        settings.logger.debug("Loaded reference candidate tensors.")
        candidates = [
            ex(tf.reshape(_, ())).numpy().decode("utf-8").split(" ") for _ in 
            [_ for _, reference in sentences][:min(30, dataset.test_raw.cardinality())]]

        settings.logger.debug("Loaded translation sample sentences for BLEU scoring.")

        tokenized_ref = dataset.tokenizers.src.tokenize(ref_tensors)
        detokenized_ref = dataset.tokenizers.src.detokenize(tokenized_ref)
        references = [r.decode("utf-8").split(" ") for r in detokenized_ref.numpy()[:len(candidates)]]
        
        
        val_bleu = corpus_bleu(references, candidates)
        settings.logger.debug("Calculated BLEU score: {}.".format(val_bleu))
        input()

    except tf.errors.ResourceExhaustedError:
        settings.logger.warn("OOM Error. Skipping architecture.")
        val_acc = "N/A"
        val_loss = "N/A"
        val_bleu = "N/A"
    
    return {
       "val_loss": val_loss,
       "val_acc": val_acc,
       "val_bleu": val_bleu}
    

if __name__ == "__main__":
    pass