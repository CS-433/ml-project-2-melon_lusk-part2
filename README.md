# Class project 2 | Road Segmentation

This is our submission for the second project of "CS-433: Machine Learning" where we designed an Image segmentation model using Deep Learning.

## Note on the used hardware

At the beginning of the project, we tried running the different code on our machines locally; however, because of limitations in computational power, it was infeasable for most of us, as it would take more than an hour to train a single step of some models.  
As such, we made use of the free [Google Colab](https://colab.research.google.com) which permitted us to make use of a GPU (an NVIDIA Tesla K80 GPU, to be exact) as to make our code go faster. As such, be weary that unless you have installed the GPU version of tensorflow, and have a powerful GPU, the code may take a very long time to execute (it takes between one and two hours on the hardware available on the free version of Google Colab).  

## Problem description

In this project, we were provided a set of satellite images acquired from Google Maps, along with the groundtruth images associated with these images. In the latter, each pixel is given one of two colors: white if the pixel in the corresponding aerial image happens to be on a road, and black otherwise. Below we show an example of an image and its groundtruth right next to it.

![alt text](https://i.imgur.com/AzjLs5M.png)
![alt text](https://i.imgur.com/YZT56cx.png)

Our task was to create a model that could generate grountruths for 50 new aerial images. 
## Data extraction

Before running any file, please extract the train and test directories in the "data" folder. Simply right click on both zip and click "extract here".

## The 'src' folder

All code, may it be scripts or notebooks, are in the 'src' folder.

### Model building
In order to build the model, we had to choose between multiple libraries (Sklearn, Pytorch, Tensorflow), and we settled on Tensorflow as we found it to be the best to learn for this project.    
We first tried to build a [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network) by basing ourselves on the given code: we would take small patches of the images (16x16, 32x32, etc...) and try to predict if this patch contains a road or not. The mask is then created from the given label (entirely black if no road, or white).    
This strategy didn't prove itself to be efficient however, and thus we turned to another solution by implementing a [Unet](https://en.wikipedia.org/wiki/U-Net) which is particularily adapted for image segmentation problems. In this case, we feed the unet the whole image and its groundtruth. This approach proved itself to be more efficient.    
The most notable models can be found in the "models" subfolder of "src"; the Unet architecture can be seen here : https://i.imgur.com/eJ0Tscx.png .

### Running the code
There are multiple scripts that can be run in the 'src' folder: if one wishes to retrain the models, they can either run the 'train_CNN.py' or the 'train_unet.py', which will create the corresponding model and save it in the 'models' folder. 
In order to generate the predictions made by our best model, as well as the corresponding CSV file that can be uploaded to [AIcrowd](https://www.aicrowd.com), simply run the "run.py" file, and it will generate a folder in which you can see the predictions made, as well as the CSV. 
Otherwise, if the predictions folder already exists, simply run "submission_creation.py" to get the corresponding CSV.
