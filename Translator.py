import os
import re

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

    def load_dataset(self, path):
        X_raw, y_raw = preprocess.load_sentences(path)

        X_sentences = preprocess.clean_sentences(X_raw)
        y_sentences = preprocess.clean_sentences(y_raw)

        self.X_tkn.fit_on_texts(X_sentences)
        self.X_train = self.X_tkn.texts_to_sequences(X_sentences)

        self.y_tkn.fit_on_texts(y_sentences)
        self.y_train = self.y_tkn.texts_to_sequences(y_sentences)

        self.X_train = pad_sequences(self.X_train, 
                                     maxlen=self.hyperparameters.max_sentence_length)
        self.y_train = pad_sequences(self.y_train,
                                     maxlen=self.hyperparameters.max_sentence_length)

        print(self.X_train[0:5])



if __name__ == "__main__":
    t = Translator()
    t.load_dataset("Datasets/Shakespeare/shakespeare.csv")
