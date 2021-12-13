import tensorflow as tf
import tensorflow.keras as keras
from keras import layers, models, datasets,regularizers, Model, Input
from keras.layers import concatenate, Conv2D, Conv2DTranspose, BatchNormalization, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from cost_functions import *

def build_CNN_model(nbr_filters  = 64):
    model = keras.Sequential()
    model.add(layers.RandomZoom(-0.3))
    model.add(layers.Conv2D(nbr_filters, (1, 1), activation='elu', padding = "same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(nbr_filters*2, (1, 1), activation='elu', padding = "same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(nbr_filters*4, (1, 1), activation='elu', padding = "same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(nbr_filters*8, (1, 1), activation='elu', padding = "same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(2))
    model.add(layers.Softmax())
    return model


def train_CNN(train_data, train_labels, epochs = 50, nbr_filters = 64):
    model = build_CNN_model(train_data,train_labels, nbr_filters)
    model.compile(optimizer='adam',
            loss= tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=[f1_score, 'accuracy'])
    model.fit(train_data, train_labels, epochs=epochs, validation_split = 0.1)
    return model