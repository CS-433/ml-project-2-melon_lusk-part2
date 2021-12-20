from helpers_unet import *
from helpers_images_prototype import *
from cost_functions import *

import tensorflow as tf

"""
When running this file, it will:
    1. Load the U-net model which performed the best
    2. Precict the masks of the test images, and store these in a "predictions" folder, so we can see them
    3. Create the corresponding CSV submission file that can be uploaded to the AIcrowd website.
"""

def main():
    #load unet model
    model = tf.keras.models.load_model('./models/model_unet.h5', custom_objects = {'dice_coef': dice_coef, 'soft_dice_loss':soft_dice_loss})
    create_submission(model)
if __name__ == '__main__':
    main()