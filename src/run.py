from helpers_unet import *
from helpers_images import *
from cost_functions import *

import tensorflow as tf

def main():
    #load model
    model = tf.keras.models.load_model('./models/model_unet.h5', custom_objects = {'dice_coef': dice_coef, 'soft_dice_loss':soft_dice_loss})
    create_submission(model)
if __name__ == '__main__':
    main()