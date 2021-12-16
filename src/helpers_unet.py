import pathlib
import os
import numpy as np


import tensorflow as tf
import tensorflow.keras as keras
from keras import layers, models, datasets,regularizers, Model, Input
from keras.layers import concatenate, Conv2D, Conv2DTranspose, BatchNormalization, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from cost_functions import *
from submission_creation import *
from helpers_images import *

PATH_TEST_IMAGES = "../data/test_set_images/"
PREDICTIONS_PATH = "./predictions/"
IMG_PATCH_SIZE = 400
####################################################################################
# Helpers to create and train the model
####################################################################################

def encoding_block(input_, nbr_filters = 16, pooling_ = True, dropout = True, dropout_rate = 0.1):
    """ One encoding block of the Unet: it's composed of two convolution layers (with their results being normalized), an optional dropout layer after the first convolution, and an optional pooling layer at the end of the block.
    """
    conv = Conv2D(nbr_filters, (3, 3), activation='elu',kernel_regularizer=regularizers.l2(0.0001), kernel_initializer='he_normal', padding='same') (input_)
    conv = BatchNormalization()(conv)
    if dropout:
        conv = Dropout(dropout_rate) (conv)
    conv = Conv2D(nbr_filters, (3, 3), activation='elu',kernel_regularizer=regularizers.l2(0.0001), kernel_initializer='he_normal', padding='same') (conv)
    conv = BatchNormalization()(conv)
    if pooling_:
        pooling = MaxPooling2D((2,2))(conv)
        return conv, pooling
    return conv


def decoding_block(conv, previous_conv,nbr_filters = 16, dropout = True, dropout_rate = 0.1):
    """ One decoding block of the Unet: it's composed one transposed convoltion layer, followed by two convolution layers (with their results being normalized), with an optional dropout layer after the first convolution.
    """
    upsample = Conv2DTranspose(nbr_filters, (2, 2),kernel_regularizer=regularizers.l2(0.01), strides=(2, 2), padding='same') (conv)
    upsample = concatenate([upsample, previous_conv])
    conv = Conv2D(nbr_filters, (3, 3), activation='elu',kernel_regularizer=regularizers.l2(0.0001), kernel_initializer='he_normal', padding='same') (upsample)
    conv = BatchNormalization() (conv)
    if dropout:
        conv = Dropout(dropout_rate) (conv)
    conv = Conv2D(nbr_filters, (3, 3), activation='elu',kernel_regularizer=regularizers.l2(0.0001), kernel_initializer='he_normal', padding='same') (conv)
    conv = BatchNormalization() (conv)
    return conv


def create_custom_unet(nbr_filters = 16, dropout = True,dropout_rate=0.1):
    """ Creates a custom unet using encoding and decoding blocks as defined earlier
    """
    inputs = Input((IMG_PATCH_SIZE, IMG_PATCH_SIZE, 3))
    conv1, pooling1 = encoding_block(inputs, nbr_filters, True, dropout,dropout_rate)
    conv2, pooling2 = encoding_block(pooling1, nbr_filters*2, True,dropout,dropout_rate)
    conv3, pooling3 = encoding_block(pooling2, nbr_filters*4, True,dropout,dropout_rate)
    conv4, pooling4 = encoding_block(pooling3, nbr_filters*8, True,dropout,dropout_rate)
    conv5 = encoding_block(pooling4, nbr_filters*16,False, dropout,dropout_rate)

    conv6 = decoding_block(conv5, conv4, nbr_filters*8, dropout,dropout_rate)
    conv7 = decoding_block(conv6, conv3, nbr_filters*4, dropout,dropout_rate)
    conv8 = decoding_block(conv7, conv2, nbr_filters*2, dropout,dropout_rate)
    conv9 = decoding_block(conv8, conv1, nbr_filters, dropout,dropout_rate)
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (conv9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


def train_unet(train_data, train_labels,epochs = 200, nbr_filters = 16, dropout = True,dropout_rate=0.1):
    """ Function used to create and train the unet model. Notice that we define a learning rate reducer and an early stopper; the goal of these
        two objects is, respectively:
        - to reduce the model's learning rate (beginning at 0.001) after 4 iterations if the validation loss doesn't go lower, as a way to try a maybe better learning rate
        - to stop the model from overfitting if after 5 iterations, the validation loss doesn't get lower. It also restores the weights at the iteration during which the validation loss was lowest
        We use a batch size of 8 , and the loss function is defined as to maximize the dice coefficient.
    """
    earlystopper = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 5,
                          verbose = 1,
                          mode = "min",
                          restore_best_weights = True)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               mode="min",
                               min_delta=1e-4)
    model = create_custom_unet(nbr_filters,dropout,dropout_rate)
    model.compile(optimizer='adam',
            loss=soft_dice_loss,
          metrics=[dice_coef])
    history = model.fit(train_data, train_labels, epochs=epochs, validation_split = 0.01, batch_size = 8,callbacks = [ earlystopper, lr_reducer])
    return model



####################################################################################
# Helpers create a submission
####################################################################################

def predict(model,all_images_arrays, k):
    """
    For a given image, will predict both the overlays and the mask; since our unet model works with 400x400 pictures and the
    target image is 608x608, we take four 400x400 patches of the image such as we cover all of it, and predict the mask of each of these four
    patches. 
    Some rows and columns overlap between the patches, but it is okay as we assume that either patch predicted the area in common correctly (and the 
    assumption turns out to be right).
    We then put these patches back together to reconstruct the whole image 
    """
    
    img = all_images_arrays[k]
    extract_1 = img[:400, :400]
    extract_2 = img[:400, -400:]
    extract_3 = img[-400:, :400]
    extract_4 = img[-400:, -400:]
    output_prediction = []
    output_prediction.append(model(np.expand_dims(extract_1 , axis=0)).numpy() )
    output_prediction.append(model(np.expand_dims(extract_2 , axis=0)).numpy() )
    output_prediction.append(model(np.expand_dims(extract_3 , axis=0)).numpy() )
    output_prediction.append(model(np.expand_dims(extract_4 , axis=0)).numpy() )

    output_prediction[0] = output_prediction[0].reshape(400,400)
    output_prediction[1] = output_prediction[1].reshape(400,400)
    output_prediction[2] = output_prediction[2].reshape(400,400)
    output_prediction[3] = output_prediction[3].reshape(400,400)
    gt_img = np.zeros([img.shape[0], img.shape[1]])
    gt_img[:400,:400] = output_prediction[0]
    gt_img[:400, -400:] = output_prediction[1]
    gt_img[-400: , :400] = output_prediction[2]
    gt_img[-400: , -400:] = output_prediction[3]
    oimg = make_img_overlay(img, gt_img)
    gt_img8 = img_float_to_uint8(gt_img)          
    gt_img_3c = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    gt_img_3c[:, :, 0] = gt_img8
    gt_img_3c[:, :, 1] = gt_img8
    gt_img_3c[:, :, 2] = gt_img8
    overlay = make_img_overlay(img, gt_img)
    Image.fromarray(gt_img_3c).save(PREDICTIONS_PATH + "prediction_image_" + str(k+1) + ".png")
    oimg.save(PREDICTIONS_PATH  + "overlay_" + str(k) + ".png")

def create_submission(model):
    directory_path = pathlib.Path(PATH_TEST_IMAGES)
    
    # Load the test images names
    all_images = []
    for i in range(1,51):
        image_in_directory = list(directory_path.glob("test_" + str(i) + "/*"))
        all_images.extend(image_in_directory)
    img_height = img_width = 608
    
    # Load the test images and normalize them
    all_images_arrays = []
    for i in range(len(all_images)):
        img = tf.keras.utils.load_img(
            all_images[i], target_size=(img_height, img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        all_images_arrays.append(img_array)
    all_images_arrays = np.array(all_images_arrays)/255.0
    
    
    # Make the predictions images
    if not os.path.isdir(PREDICTIONS_PATH):
        os.mkdir(PREDICTIONS_PATH)
    
    for i in range(len(all_images_arrays)):
        predict(model,all_images_arrays,i)
        
        
    # Create the csv
    submission_filename = 'submission.csv'
    image_filenames = []
    for i in range(1, 51):
        image_filename = PREDICTIONS_PATH + 'prediction_image_' + '%.1d' % i + '.png'
        print(image_filename)
        image_filenames.append(image_filename)
    masks_to_submission(submission_filename, *image_filenames)
    