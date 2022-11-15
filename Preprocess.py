import os
import re

import pandas as pd
import numpy as np

import keras
from tensorflow.keras.preprocessing.text import Tokenizer

def read_dataset():
    df = pd.read_csv(os.path.join(os.getcwd(), "Datasets/Shakespeare/shakespeare.csv"))
    og = df["og"].to_list()
    tran = df["t"].to_list()
    return og, tran

def add_SOSEOS(sentences):
    for i in range(len(sentences)):
        sentences[i] = "<\SOS>" + sentences[i] + "<\EOS>"
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

    return cleaned_sentences

def tokenize(sentences):
    pass

if __name__ == "__main__":
    X_raw, y_raw = read_dataset()
    
    X_sentences = clean_sentences(X_raw)
    Y_sentences = clean_sentences(y_raw)

    print(X_raw[0:5])

    Xtkn = Tokenizer()

    Xtkn.fit_on_texts(X_sentences)


    X_train = Xtkn.texts_to_sequences(X_sentences)