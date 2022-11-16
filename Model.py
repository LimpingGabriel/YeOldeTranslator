from keras import Sequential

from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras.layers import GRU
from keras.layers import Dropout
from tensorflow.keras.optimizers import Adam

class Model(Sequential):
    """description of class"""

    def __init__(self):
        super().__init__()

    def createModel(self, hyperparameters):

        if hyperparameters.modelType == "basicRNN":
            self.add(GRU(256, input_shape=hyperparameters.X_train_shape[1:], return_sequences=True))
            self.add(TimeDistributed(Dense(1024, activation="relu")))
            self.add(Dropout(0.5))
            self.add(TimeDistributed(Dense(hyperparameters.target_vocab_size, activation="softmax")))

            self.compile(optimizer=Adam(0.005),
                        loss="sparse_categorical_crossentropy",
                        metrics=["accuracy"])



            print(self.summary())
