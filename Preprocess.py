######
#HYPERPARAMETERS#
MAX_WORDS = 25 #Max words to use in any sentence for RNN

######

import os
import re

import pandas as pd
import numpy as np

import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_sentences(path):
    df = pd.read_csv(os.path.abspath(path))
    og = df["og"].to_list()
    tran = df["t"].to_list()
    return og, tran

def add_soseos(sentences):
    for i in range(len(sentences)):
        sentences[i] = "</SOS> " + sentences[i] + " </EOS>"
    return sentences

def clean_sentences(sentences):
    letters = re.compile("[^a-zA-Z ]+")
    spaces = re.compile(" +")
    cleaned_sentences = list()

    for sentence in sentences:
        cleaned_sentence = sentence.replace("—", " ")
        cleaned_sentence = letters.sub("", cleaned_sentence)
        cleaned_sentence = cleaned_sentence.lower()
        cleaned_sentence = spaces.sub(" ", cleaned_sentence)
        if cleaned_sentence[0] == " ":
            cleaned_sentence = cleaned_sentence[1:]
        if cleaned_sentence[-1] == " ":
            cleaned_sentence = cleaned_sentence[:-1]

        cleaned_sentences.append(cleaned_sentence)

    cleaned_sentences = add_soseos(cleaned_sentences)
    return cleaned_sentences

