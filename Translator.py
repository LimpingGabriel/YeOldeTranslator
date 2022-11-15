import os
import re
import logging
import time

import pandas as pd
import numpy as np

import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import preprocess
from Hyperparameters import Hyperparameters

class Translator():
    def __init__(self):
        self.X_tkn = Tokenizer()
        self.y_tkn = Tokenizer()
        self.hyperparameters = Hyperparameters()
        self.hyperparameters.max_sentence_length = 50



        #For logging
        #Figure out better ID
        self.id = time.time()
        self.log_d = {"id": str(self.id)}

        self.logger = logging.getLogger(str(self.id))
        self.handler = logging.StreamHandler()

        log_format = "%(asctime)s.%(msecs)03d %(levelname)-8s %(id)s %(message)s"
        datefmt = '%Y-%m-%d %H:%M:%S'

        self.logfmt = logging.Formatter(log_format, datefmt)
        self.handler.setFormatter(self.logfmt)
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.DEBUG)

    def load_dataset(self, path):
        X_raw, y_raw = preprocess.load_sentences(path)
        self.logger.log(logging.DEBUG, "Loaded raw sentences from dataset.", extra=self.log_d)

        X_sentences = preprocess.clean_sentences(X_raw)
        y_sentences = preprocess.clean_sentences(y_raw)
        self.logger.log(logging.DEBUG, "Cleaned sentences.", extra=self.log_d)

        self.X_tkn.fit_on_texts(X_sentences)
        self.X_train = self.X_tkn.texts_to_sequences(X_sentences)
        self.logger.log(logging.DEBUG, "Tokenized input sentences.", extra=self.log_d)

        self.y_tkn.fit_on_texts(y_sentences)
        self.y_train = self.y_tkn.texts_to_sequences(y_sentences)
        self.logger.log(logging.DEBUG, "Tokenized output sentences.", extra=self.log_d)

        self.X_train = pad_sequences(self.X_train, 
                                     maxlen=self.hyperparameters.max_sentence_length)
        self.logger.log(logging.DEBUG, "Padded input sentences.", extra=self.log_d)

        self.y_train = pad_sequences(self.y_train,
                                     maxlen=self.hyperparameters.max_sentence_length)
        self.logger.log(logging.DEBUG, "Padded output sentences.", extra=self.log_d)

        self.logger.log(logging.INFO, "Loaded {} dataset.".format(path), extra=self.log_d)

if __name__ == "__main__":
    t = Translator()
    t.load_dataset("Datasets/Shakespeare/shakespeare.csv")
