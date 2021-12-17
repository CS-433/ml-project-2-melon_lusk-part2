from helpers_unet import *
from helpers_images_prototype import *

PATH_TEST_IMAGES = "../data/test_set_images/"
PATH_TRAINING_IMAGES = "../data/training/"

"""
In case one would want to train the U-net model again, they would have to run this file (after modifying whatever hyperparameters they want).
The created model will be saved in the 'models' folder.

When running this on a GPU, it would take one hour at most; running it on a CPU will be many times slower.
"""


def main():
    #load data
    img_size = 400
    NUMBER_TRAINING_EXAMPLES = 100
    train_data = extract_train_data(NUMBER_TRAINING_EXAMPLES,img_size)
    train_labels =  extract_labels(training_data_directory, NUMBER_TRAINING_EXAMPLES, img_size, True)
    #define training params and train
    epochs = 50
    nbr_filters = 16
    dropout = True
    dropout_rate = 0.1
    model = train_unet(train_data, train_labels,epochs, nbr_filters, dropout,dropout_rate)
    #save model
    model.save("./models/model_unet.h5")
if __name__ == '__main__':
    main()