import os


from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

from Shakespeare import load_shakespeare
from GlobalSettings import GlobalSettings

settings = GlobalSettings()

class TextDataset(object):
    """Description here"""
    def __init__(self, dstype, dirname=""):
        self.dstype = dstype
        self.dirname = dirname
        self.src_vocab = None
        self.tar_vocab = None
        self.raw_sentences = None

        settings.logger.info("Created {} TextDataset.".format(self.dstype))

    def load_data(self):
        if self.dstype == "SNShakespeare":
            self.raw_sentences = load_shakespeare()
        
        settings.logger.info("Loaded {} dataset.".format(self.dstype))
    
    def write_vocab_file(self, fpath, vocab):
        with open(fpath, "w") as f:
            for token in vocab:
                print(token, file=f)

    def generate_vocabulary(self):
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

    def build_tokenizer(self):
        if (not os.path.isfile("{}src_vocab.txt".format(self.dirname))) or (not os.path.isfile("{}tar_vocab.txt".format(self.dirname))):
            settings.logger.warn("BERT Vocabulary file(s) missing.")
            if (self.raw_sentences is None):
                settings.logger.warn("{} Dataset has not been loaded.".format(self.dstype))
                self.load_data()
            self.generate_vocabulary()
        else:
            settings.logger.debug("Found vocabulary files.")


        ##
        
