import re
import os
import pathlib

import tensorflow as tf
import tensorflow_text as tf_text

from GlobalSettings import GlobalSettings
settings = GlobalSettings()

class CustomTokenizer(tf.Module):
    """
    Create a custom tokenizer and some handy functions for easily tokenizing/detokenizing.
    """
    def __init__(self, reserved_tokens, vocab_path):
        # Create BERT tokenizer, and load saved vocabulary.
        self.tokenizer = tf_text.BertTokenizer(vocab_path, lower_case=True)
        self._reserved_tokens = reserved_tokens
        self._vocab_path = tf.saved_model.Asset(vocab_path)

        vocab = pathlib.Path(vocab_path).read_text().splitlines()
        self.vocab = tf.Variable(vocab)

        settings.logger.debug("Loaded BERT tokenizer file {}.".format(vocab_path))

        # These get_concrete_functions trace the tensorflow functions defined below
        # Which allows for faster execution in the future.
        # https://www.tensorflow.org/guide/function
        self.tokenize.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string)
            )
        settings.logger.debug("Traced tokenize method.")
        

        self.detokenize.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64)
            )
        self.detokenize.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64)
            )
        settings.logger.debug("Traced detokenize method.")

        self.lookup.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64)
            )
        self.lookup.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64)
            )
        settings.logger.debug("Traced lookup method.")

        self.get_vocab_size.get_concrete_function()
        self.get_vocab_path.get_concrete_function()
        self.get_reserved_tokens.get_concrete_function()
        settings.logger.debug("Traced getter methods.")

        settings.logger.info("Created {} tokenizer.".format(vocab_path))

    @tf.function
    def tokenize(self, strings):
        enc = self.tokenizer.tokenize(strings)
        # Merges word and word pieces axes 
        enc = enc.merge_dims(-2, -1)
        enc = add_start_end(enc)
        settings.logger.debug("Tokenized strings.")
        return enc

    @tf.function
    def detokenize(self, tokenized):
        words = self.tokenizer.detokenize(tokenized)
        settings.logger.debug("Detokenized tokens.")
        return cleanup_text(self._reserved_tokens, words)

    @tf.function
    def lookup(self, token_ids):
        settings.logger.debug("Completed token lookup.")
        return tf.gather(self.vocab, token_ids)

    @tf.function
    def get_vocab_size(self):
        settings.logger.debug("Retrieved vocabulary size.")
        return tf.shape(self.vocab)[0]

    @tf.function
    def get_vocab_path(self):
        settings.logger.debug("Retrieved vocabulary path.")
        return self._vocab_path

    @tf.function
    def get_reserved_tokens(self):
        settings.logger.debug("Retrieved reserved tokens.")
        return tf.constant(self._reserved_tokens)







# Specify the tokens
START = tf.argmax(tf.constant(settings.bert_reserved_tokens) == "[START]")
END = tf.argmax(tf.constant(settings.bert_reserved_tokens) == "[END]")

def add_start_end(ragged):
    """Returns tensor with start and end tokens added on."""
    count = ragged.bounding_shape()[0]
    starts = tf.fill([count, 1], START)
    ends = tf.fill([count, 1], END)
    settings.logger.debug("Added start and end tokens.")
    return tf.concat([starts, ragged, ends], axis=1)

def cleanup_text(reserved_tokens, token_text):
    """Converts tokenized text into a fully human-readable format."""
    bad_tokens = [re.escape(tok) for tok in reserved_tokens if tok != "[UNK]"]
    bad_tokens_re = "|".join(bad_tokens)

    bad_cells = tf.strings.regex_full_match(token_text, bad_tokens_re)
    result = tf.ragged.boolean_mask(token_text, ~bad_cells)

    settings.logger.debug("Removed bad tokens.")

    result = tf.strings.reduce_join(result, separator=" ", axis=-1)

    return result