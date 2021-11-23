import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image

import code

#import tensorflow.python.platform

import numpy
import tensorflow as tf
import tensorflow.keras as keras
from keras import layers, models, datasets
from helpers import * 
#import tensorflow.compat.v1 as tf
#tf.disable_eager_execution()


"""

What to do:

- Load the data, transform it into a dataset
- Split into train, validation, test
- Data augmentation
- Create model (auto tunining?)
- Compile model
- Fit
- Predict

"""


RATIO_VALIDATION = 0.2
NUMBER_TRAINING_EXAMPLES = 100
NUMBER_VALIDATION_EXAMPLES = RATIO_VALIDATION * NUMBER_TRAINING_EXAMPLES
NUMBER_TEST_EXAMPLES = 50
def main():
    
    training_data_directory = "../data/training/"
    test_data_directory = "../data/test_set_images/"
    
    train_data, train_labels = extract_train_data(NUMBER_TRAINING_EXAMPLES), extract_labels(training_data_directory, NUMBER_TRAINING_EXAMPLES)
    epochs = 100
    
    print(train_data.shape)
    print(train_labels.shape)
    model = keras.Sequential()
    model.add(layers.Conv2D(16, (2, 2), activation='relu', input_shape=(16, 16, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (2, 2), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    print(model.summary())
    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(1))
    
    print(model.summary())
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    history = history = model.fit(train_data, train_labels, epochs=10)
main()