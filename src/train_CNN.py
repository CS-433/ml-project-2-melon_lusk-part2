from helpers_CNN import *
from helpers_images import *
import os
PATH_TEST_IMAGES = "../data/test_set_images/"
PATH_TRAINING_IMAGES = "../data/training/"


"""
In case one would want to train the CNN model again, they would have to run this file (after modifying whatever hyperparameters they want).
The created model will be saved in the 'models' folder.
"""

def main():
    #Running on GPU ends up in an OOM even on a 3080, so we force execution on a CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    #load data
    img_patch_size = 16
    NUMBER_TRAINING_EXAMPLES = 100
    train_data = extract_train_data(NUMBER_TRAINING_EXAMPLES,img_patch_size)
    train_labels =  extract_labels(training_data_directory, NUMBER_TRAINING_EXAMPLES,img_patch_size, False)
    print(train_data.shape)
    print(train_labels.shape)
    #define training params and train
    epochs = 50
    nbr_filters = 64
    model = train_CNN(train_data, train_labels, epochs, nbr_filters = 64)
    #save model
    model.save("./models/model_CNN.h5")
if __name__ == '__main__':
    main()