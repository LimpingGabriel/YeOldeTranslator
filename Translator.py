import os
import re
import logging
import time
import pickle

import pandas as pd
import numpy as np

import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import preprocess
from Hyperparameters import Hyperparameters
from Model import Model

class Translator():
    def __init__(self):
        self.hyperparameters = Hyperparameters()
        self.X_tkn = Tokenizer(num_words=self.hyperparameters.source_vocab_size)
        self.y_tkn = Tokenizer(num_words=self.hyperparameters.target_vocab_size)
        self.hyperparameters.max_sentence_length = 15
        self.hyperparameters.modelType = "basicRNN"

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

        self.X_train = np.array(pad_sequences(self.X_train, 
                                     maxlen=self.hyperparameters.max_sentence_length))

        self.y_train = np.array(pad_sequences(self.y_train,
                                     maxlen=self.hyperparameters.max_sentence_length))
        self.y_train = self.y_train.reshape(*self.y_train.shape, 1)

        self.X_train = self.X_train.reshape((-1, self.y_train.shape[-2], 1))
        self.hyperparameters.X_train_shape = self.X_train.shape
        self.hyperparameters.y_train_shape = self.y_train.shape


        self.logger.log(logging.DEBUG, "Padded input sentences.", extra=self.log_d)

        self.logger.log(logging.DEBUG, "Padded output sentences.", extra=self.log_d)
        
        self.logger.log(logging.INFO, "Loaded {} dataset.".format(path), extra=self.log_d)

    def createModel(self):
        self.model = Model()
        self.model.createModel(self.hyperparameters)

    def train(self):
        self.model.fit(
            self.X_train, 
            self.y_train, 
            epochs=20, 
            batch_size=128, 
            verbose=2,
            validation_split=0.1)
        #self.model.save(time.time())

    def predict(self, sentence):
        clean_sentence = preprocess.clean_sentences([sentence])
        enc_sentence = np.array(
            pad_sequences(
                self.X_tkn.texts_to_sequences(sentence),
                maxlen=self.hyperparameters.max_sentence_length))
        enc_sentence = enc_sentence.reshape((-1, self.hyperparameters.y_train_shape[-2], 1))

        labels = self.model.predict(enc_sentence)[0]

        print(preprocess.labels_to_text(labels, self.y_tkn))

    def save(self):
        self.hyperparameters.X_tkn = self.X_tkn
        self.hyperparameters.y_tkn = self.y_tkn
        self.model.save(str(self.id))
        pickle.dump(self.hyperparameters, open(os.path.abspath(str(self.id) + "/" + str(self.id) + ".p"), "wb"))

    def loadModel(self, id):
        self.id = float(id)
        self.hyperparameters = pickle.load(open(os.path.abspath(id + "/" + str(self.id) + ".p"), "rb"))
        self.X_tkn = self.hyperparameters.X_tkn
        self.y_tkn = self.hyperparameters.y_tkn

        self.model = keras.models.load_model(str(self.id))

if __name__ == "__main__":
    """
    t = Translator()
    t.loadModel("1668563942.3499851")
    t.predict("Hello there!")
    """
    t = Translator()
    t.load_dataset("Datasets/Shakespeare/shakespeare.csv")
    t.createModel()
    t.train()
    t.predict("But what's the matter?")
    t.predict("He that hath lost her too; so is the queen, That most desired the match;")
    t.predict("but not a courtier, Although they wear their faces to the bent Of the king's look's, hath a heart...")
    t.predict("He that hath miss'd the princess is a thing Too bad for bad report:")
    t.predict("You speak him far.")
    t.save()
