import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu

from TextDataset import *

NUM_TESTING = 1000
MODELS = [
    "transformer_batch_size-32__num_layers-6__d_model-128__dff-512__num_heads-8__dropout-0.3__epochs-50_",
    "transformer_batch_size-64__num_layers-4__d_model-128__dff-256__num_heads-8__dropout-0.1__epochs-50_",
    "transformer_batch_size-64__num_layers-4__d_model-128__dff-256__num_heads-8__dropout-0.3__epochs-50_",
    "transformer_batch_size-64__num_layers-4__d_model-128__dff-256__num_heads-8__dropout-0.3__epochs-200_",
    ]

if __name__ == "__main__":
    dataset = TextDataset("Shakescleare", dirname="ShakescleareTokens")
    test_dataset = TextDataset("SNShakespeare", dirname="SNShakespeareTokens")

    dataset.prepare()
    test_dataset.prepare()
    
    reference_corpus = []
    candidate_corpus = []

    sentences = test_dataset.raw_sentences.shuffle(20000, seed=42).batch(1).take(test_dataset.raw_sentences.cardinality())
    settings.logger.debug("Batched {} sentences for BLEU scoring.".format(dataset.raw_sentences.cardinality()))
    
    ref_tensors = tf.concat([reference for _, reference in sentences], 0)
    settings.logger.debug("Loaded reference candidate tensors.")

    tokenized_ref = dataset.tokenizers.src.tokenize(ref_tensors)
    detokenized_ref = dataset.tokenizers.src.detokenize(tokenized_ref)

    settings.logger.debug("Tokenized reference sentences.")

    references = [[r.decode("utf-8").split(" ")] for r in detokenized_ref.numpy()[:NUM_TESTING]]

    scores = {}

    for model in MODELS:
        ex = tf.saved_model.load(model)
        settings.logger.debug("Loaded {}.".format(model))
        candidates = [
            ex(tf.reshape(_, ())).numpy().decode("utf-8").split(" ") for _ in 
            [_ for _, reference in sentences][:NUM_TESTING]]
        settings.logger.debug("Loaded {} translation sample sentences for BLEU scoring.".format(NUM_TESTING))
        bleu = {
            "1-gram": corpus_bleu(references, candidates, weights=(1, 0, 0, 0)),
            "2-gram": corpus_bleu(references, candidates, weights=(0.5, 0.5, 0, 0)),
            "3-gram": corpus_bleu(references, candidates, weights=(1/3, 1/3, 1/3, 0)),
            "4-gram": corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25))}

        print("Model {} has BLEU score of {} on {} samples of SparkNotes Shakespeare.".format(
            model, bleu["1-gram"], NUM_TESTING))
        scores[model] = bleu

    with open("BLEU Scores.txt", "a") as f:
        f.write(str(scores))
        settings.logger.info("Saved BLEU scores.")