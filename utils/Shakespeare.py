import os
import re
import io
from GlobalSettings import GlobalSettings
import tensorflow as tf
import numpy as np
import pandas as pd

settings = GlobalSettings()

def load_snshakespeare():
    """Returns tf.data.Dataset from shakespeare dataset."""

    original = []
    modern = []

    fdir = "Datasets\Shakespeare\sparknotes\merged\\"

    for fname in os.listdir(os.path.abspath(fdir)):
        if re.search("(original\.snt\.aligned)", fname):
            with io.open(os.path.abspath(fdir + fname), "r", encoding="utf-8") as f:
                settings.logger.debug("Loaded {}".format(fname))
                lines = f.readlines()
                for line in lines:
                    original.append(re.sub('(?<! )(?=[.,!?()])|(?<=[.,!?()])(?! )', r' ', line))
        
        elif re.search("(modern\.snt\.aligned)", fname):
            with io.open(os.path.abspath(fdir + fname), "r", encoding="utf-8") as f:
                settings.logger.debug("Loaded {}".format(fname))
                lines = f.readlines()
                for line in lines:
                    modern.append(line)

    if len(original) != len(modern):
        settings.logger.critical("Original length {} does not match modern length {}!".format(len(original), len(modern)))
        raise AssertionError()

    data = tf.data.Dataset.from_tensor_slices((modern, original))
    settings.logger.info("Loaded {} samples.".format(len(original)))
    return data

def load_shakescleare():
    """Returns tf.data.Dataset from shakespeare dataset."""

    fdir = "Datasets\Shakespeare\shakespeare.csv"
    df = pd.read_csv(fdir)

    original = df["og"].tolist()
    modern = df["t"].tolist()
    settings.logger.debug("Loaded {}.".format(fdir))
    
    if len(original) != len(modern):
        settings.logger.critical("Original length {} does not match modern length {}!".format(len(original), len(modern)))
        raise AssertionError()

    data = tf.data.Dataset.from_tensor_slices((modern, original))
    settings.logger.info("Loaded {} samples.".format(len(original)))
    return data

if __name__ == "__main__":
    pass
    #load_snshakespeare()