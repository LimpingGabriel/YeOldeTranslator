import os


from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
import tensorflow_text as tf_text
import tensorflow as tf

from Shakespeare import *
from GlobalSettings import GlobalSettings
from CustomTokenizer import CustomTokenizer

settings = GlobalSettings()

class TextDataset(object):
    """Description here"""
    def __init__(self, dstype, dirname=""):
        self.dstype = dstype
        self.dirname = dirname
        self.src_vocab = None
        self.tar_vocab = None
        self.raw_sentences = None
        self.tokenizers = None

        settings.logger.info("Created {} TextDataset.".format(self.dstype))

    def load_data(self):
        if self.dstype == "SNShakespeare":
            self.raw_sentences = load_snshakespeare().shuffle(settings.BUFFER_SIZE)
        if self.dstype == "Shakescleare":
            self.raw_sentences = load_shakescleare().shuffle(settings.BUFFER_SIZE)
        
        settings.logger.info("Loaded {} dataset.".format(self.dstype))
    
    def write_vocab_file(self, fpath, vocab):
        with open(fpath, "w") as f:
            for token in vocab:
                print(token, file=f)

    def generate_vocabulary(self):
        if (self.raw_sentences is None):
            settings.logger.warn("{} Dataset has not been loaded.".format(self.dstype))
            self.load_data()
        train_tar = self.raw_sentences.map(lambda src, en: en)
        train_src = self.raw_sentences.map(lambda tar, en: tar)
        settings.logger.debug("Loaded source and target tensors for BERT.")

        bert_vocab_args = dict(
            vocab_size = settings.bert_vocab_size,
            reserved_tokens = settings.bert_reserved_tokens,
            bert_tokenizer_params = settings.bert_tokenizer_params,
            learn_params={})

        self.src_vocab = bert_vocab.bert_vocab_from_dataset(
            train_src.batch(1000).prefetch(2),
            **bert_vocab_args)
        settings.logger.debug("Created source vocabulary.")
        
        self.tar_vocab = bert_vocab.bert_vocab_from_dataset(
            train_tar.batch(1000).prefetch(2),
            **bert_vocab_args)
        settings.logger.debug("Created target vocabulary.")

        settings.logger.info("Created vocabulary.")

        self.write_vocab_file(os.path.abspath("{}src_vocab.txt".format(self.dirname)), self.src_vocab)
        settings.logger.debug("Wrote source vocabulary file.")

        self.write_vocab_file(os.path.abspath("{}tar_vocab.txt".format(self.dirname)), self.tar_vocab)
        settings.logger.debug("Wrote target vocabulary file.")

        settings.logger.info("Created vocabulary files.")

    def build_tokenizers(self):
        if (not os.path.isfile("{}src_vocab.txt".format(self.dirname))) or (not os.path.isfile("{}tar_vocab.txt".format(self.dirname))):
            settings.logger.warn("BERT Vocabulary file(s) missing.")
            self.generate_vocabulary()
        else:
            settings.logger.debug("Found vocabulary files.")

        src_tokenizer = tf_text.BertTokenizer(os.path.abspath("{}src_vocab.txt".format(self.dirname)), **settings.bert_tokenizer_params)
        settings.logger.debug("Created source tokenizer.")

        tar_tokenizer = tf_text.BertTokenizer(os.path.abspath("{}tar_vocab.txt".format(self.dirname)), **settings.bert_tokenizer_params)
        settings.logger.debug("Created target tokenizer.")

        self.tokenizers = tf.Module()
        self.tokenizers.src = CustomTokenizer(settings.bert_reserved_tokens, os.path.abspath("{}src_vocab.txt".format(self.dirname)))
        self.tokenizers.tar = CustomTokenizer(settings.bert_reserved_tokens, os.path.abspath("{}tar_vocab.txt".format(self.dirname)))
        
        settings.logger.debug("Created tokenizer module.")

        model_name = "{}tokenizer".format(self.dirname)

        tf.saved_model.save(self.tokenizers, model_name)
        settings.logger.info("Built and saved tokenizer.")

    def load_tokenizers(self):
        if self.tokenizers == None:
            try:
                self.tokenizers = tf.saved_model.load("{}tokenizer".format(self.dirname))
            except (OSError, IOError) as e:
                settings.logger.warn("No tokenizer file found at {}.".format(self.dirname))
                self.build_tokenizers()

        settings.logger.info("Loaded tokenizer.")

    def prepare_batch(self, src, tar):
        src = self.tokenizers.src.tokenize(src)
        src = src[:, :settings.MAX_TOKENS]
        src = src.to_tensor()

        tar = self.tokenizers.tar.tokenize(tar)
        tar = tar[:, :(settings.MAX_TOKENS+1)]
        tar_inputs = tar[:, :-1].to_tensor()
        tar_labels = tar[:, 1:].to_tensor()

        return (src, tar_inputs), tar_labels

    def make_split(self, train_test_split=1):
        shuffled_sentences = self.raw_sentences.shuffle(settings.BUFFER_SIZE, seed=42)

        train_size = int(train_test_split * shuffled_sentences.cardinality().numpy())
        val_size = int(0.5 * (1 - train_test_split) * shuffled_sentences.cardinality().numpy())
        test_size = int(0.5 * (1 - train_test_split) * shuffled_sentences.cardinality().numpy())


        self.train_raw = shuffled_sentences.take(train_size)
        self.test_raw = shuffled_sentences.skip(train_size)
        self.valid_raw = self.test_raw.skip(test_size)
        self.test_raw = self.test_raw.take(test_size)

        

    def make_batches(self, batch_size):

        return ((
            self.train_raw
            .shuffle(settings.BUFFER_SIZE)
            .batch(batch_size)
            .map(self.prepare_batch, tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
            ),
            
            (
            self.valid_raw
            .shuffle(settings.BUFFER_SIZE)
            .batch(settings.BATCH_SIZE)
            .map(self.prepare_batch, tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
            ))


    def prepare(self):
        """Used to fully load the dataset ready-to-go for training."""
        self.load_data()
        self.load_tokenizers()
