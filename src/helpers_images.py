import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image
import code
import tensorflow.python.platform
import numpy
import tensorflow as tf

training_data_directory = "../data/training/"
test_data_directory = "../data/test_set_images/"
PIXEL_DEPTH = 255
IMG_PATCH_SIZE = 400
IMG_SIZE = 400

# Extract patches from a given image
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

def extract_data_from_image(img_array):
    IMG_WIDTH = img_array[0].shape[0]
    IMG_HEIGHT = img_array[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)
    
    img_patches = img_crop(img_array,IMG_PATCH_SIZE,IMG_PATCH_SIZE)
    data = [img_patches[i] for i in range(len(img_patches))]
    return data

def extract_data_from_directory(directory_name, file_basename, num_images, training = True):
    """
    Function used to extract image data from a directory. Most notably used  in the function "extract_train_data".
    """
    
    imgs = []
    for i in range(1, num_images+1):
        imageid = file_basename + "_%.3d"%i
        #imageid = "satImage_%.3d" % i
        image_filename = directory_name + "images/" + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            #img = mpimg.imread(image_filename)
            image = Image.open(image_filename)
            image = image.resize((IMG_SIZE,IMG_SIZE), Image.ANTIALIAS)
            image = (numpy.array(image.getdata())/255.0).reshape(IMG_SIZE,IMG_SIZE,3)
            imgs.append(image)
        else:
            print('File ' + image_filename + ' does not exist')
    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)
    
    """
    From this point on, we carry out multiple data augmentations as a way to not only have more data but also to prevent overfitting.
    We apply the transformations successively on the original data.
    """
    
    # DATA AUGMENTATION: BEGIN
    
    for i in range(num_images):
        imgs.append(tf.image.flip_left_right(imgs[i])) #horizontal flip of the image
    for i in range(num_images):
        transformation = tf.image.rot90(imgs[i]) #90 degrees rotation of the image
        imgs.append(transformation)
        transformation = tf.image.rot90(transformation) #180 degrees rotation of the image
        imgs.append(transformation)
        imgs.append(tf.image.rot90(transformation)) # 270 degrees rotation of the image
    for i in range(num_images):
        seed = (i, 0)
        imgs.append(tf.image.stateless_random_brightness(imgs[i],0.2,seed)) # adjust the brightness of the image
    for i in range(num_images):
        seed = (i, 0)
        imgs.append(tf.image.stateless_random_contrast(imgs[i], lower=0.1, upper=0.9, seed=seed)) #adjust the contrast of the image
    for i in range(num_images):
        seed = (i, 0)
        imgs.append(tf.image.stateless_random_saturation(imgs[i], lower=0.1, upper=0.9, seed=seed)) # adjust the saturation of the image
    for i in range(num_images):
        imgs.append(tf.image.random_jpeg_quality(imgs[i], 75, 95, seed=i)) # randomly change the image's quality; a way to introduce noise in the data
    
    # DATA AUGMENTATION: END
    
    num_images = len(imgs)
    img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
    return numpy.asarray(data)
            
def extract_train_data(num_images):
    return extract_data_from_directory(training_data_directory,"satImage", num_images, True)

# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
    if df > foreground_threshold:  # road
        return [0, 1]
    else:  # bgrd
        return [1, 0]


# Extract label images
def extract_labels(filename, num_images, unet = False):
    """
    Extract the labels into a 1-hot matrix [image index, label index] in the case of a CNN. Otherwise, return the groundtruths of the training
    data after applying data augmentation.
    """
    gt_imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + "groundtruth/" + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            #img = mpimg.imread(image_filename)
            image = Image.open(image_filename)
            image = image.resize((IMG_SIZE,IMG_SIZE), Image.ANTIALIAS)
            img = (numpy.array(image.getdata())/255.0).reshape(IMG_SIZE,IMG_SIZE)
            
            if unet:
                img[img > 0] = 1.0 #transform it to black and white
            
            gt_imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')
    num_images = len(gt_imgs)
    """
    In this part of the code, we apply the corresponding data augmentations on the groundtruth so that they match the data augmentations carried out on the original data
    in the function 'extract_data_from_directory'. 
    For the transformations like contrast or saturation, we simply add the corresponding groundtruth without applying the transformation to them, 
    as it wouldn't really make sense for these kinds of augmentations.
    """
    
    # DATA AUGMENTATION : BEGIN
    
    for i in range(num_images):
        transformation = numpy.expand_dims(gt_imgs[i], axis = 2)
        transformation = tf.image.flip_left_right(transformation)  # horizontal flip of the groundtruth
        gt_imgs.append(transformation[:,:,0].numpy())
    for i in range(num_images):
        transformation = numpy.expand_dims(gt_imgs[i], axis = 2)
        transformation = tf.image.rot90(transformation)  # 90 degrees rotation of the groundtruth
        gt_imgs.append(transformation[:,:,0].numpy())
        transformation = tf.image.rot90(transformation)  # 180 degrees rotation of the groundtruth
        gt_imgs.append(transformation[:,:,0].numpy())
        gt_imgs.append(tf.image.rot90(transformation)[:,:,0].numpy())  # 270 degrees rotation of the groundtruth
    for i in range(num_images):
        gt_imgs.append(gt_imgs[i]) # groundtruth corresponding to the brightness change
    for i in range(num_images):
        gt_imgs.append(gt_imgs[i]) # groundtruth corresponding to the contrast change
    for i in range(num_images):
        gt_imgs.append(gt_imgs[i]) # groundtruth corresponding to the saturation change
    for i in range(num_images):
        gt_imgs.append(gt_imgs[i]) # groundtruth corresponding to the quality change
        
    # DATA AUGMENTATION: END
    
    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    
    if unet: #if we are using a unet, we expand the dimensions of the patches as the unet works with tensors and not matrices
        data = numpy.asarray([numpy.expand_dims(gt_patches[i][j],-1) for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    else:
        data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    if unet: # if we are using a unet, we return the data. Otherwise, we apply the value to class function which, for each patch, gives only one classification (road or foreground).
        return data 
    labels = numpy.asarray([value_to_class(numpy.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)

# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = numpy.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if labels[idx][0] > 0.5:  # bgrd
                l = 0
            else:
                l = 1
            array_labels[j:j+w, i:i+h] = l
            idx = idx + 1
    return array_labels


def img_float_to_uint8(img):
    rimg = img - numpy.min(img)
    rimg = (rimg / numpy.max(rimg) * PIXEL_DEPTH).round().astype(numpy.uint8)
    return rimg


def concatenate_images(img, gt_img):
    n_channels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if n_channels == 3:
        cimg = numpy.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = numpy.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:, :, 0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


def get_prediction(img, model):
    data = numpy.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE))
    output_prediction = model(data)
    img_prediction = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)

    return img_prediction
def get_prediction_with_groundtruth(img, model):
    img_prediction = get_prediction(img, model)
    cimg = concatenate_images(img, img_prediction)

    return cimg, img_prediction
    
    
def get_prediction_with_overlay(img, model):
    img_prediction = get_prediction(img,model)
    oimg = make_img_overlay(img, img_prediction)
    return oimg


def get_prediction_with_groundtruth_from_file(filename, image_idx, model):
    imageid = "satImage_%.3d" % image_idx
    image_filename = filename + imageid + ".png"
    img = mpimg.imread(image_filename)

    img_prediction = get_prediction(img,model)
    cimg = concatenate_images(img, img_prediction)

    return cimg