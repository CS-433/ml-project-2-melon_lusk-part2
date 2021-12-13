from helpers_unet import *
from helpers_images import *

PATH_TEST_IMAGES = "../data/test_set_images/"
PATH_TRAINING_IMAGES = "../data/training/"


def main():
    #load data
    train_data = extract_train_data(NUMBER_TRAINING_EXAMPLES)
    train_labels =  extract_labels(training_data_directory, NUMBER_TRAINING_EXAMPLES,True)
    #define training params and train
    epochs = 200
    nbr_filters = 16
    dropout = True
    dropout_rate = 0.1
    model = train_unet(train_data, train_labels,epochs, nbr_filters, dropout,dropout_rate)
    #save model
    model.save("./models/model_unet.h5")
if __name__ == '__main__':
    main()