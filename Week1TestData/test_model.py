"""

This is for morphological classification of galaxies by CNN,
By Kenji Bekki, on 2017/11/15
Revised on 2020/2/14 (Nair & Abraham 2010)
Refactored by Patrick Roe, on 2022/07/29
For test only.

"""

import numpy as np

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import model_from_json

# Filenames for Data
TESTING_IMAGES_PATH = "nam_images_test.dat"
TESTING_LABELS_PATH = "nam_labels_test.dat"
OUTFILE_PATH = "test.out"

# Filenames for Model
MODEL_PATH = "model.json"
MODEL_WEIGHTS_PATH = "model.h5"

# Image Data specifications
IMG_HEIGHT = 50
IMG_WIDTH = 50
NUM_TESTING_IMAGES = 2681
NUM_CLASSES = 3

def loadImages():

    # Converter to subtract 1 from all labels
    label_converter = lambda x: int(x) - 1;     

    # Load & Reshape Testing Labels
    testing_labels = np.genfromtxt(TESTING_LABELS_PATH, dtype=np.uint8, converters={0: label_converter})
    testing_labels = np.reshape(testing_labels, (NUM_TESTING_IMAGES, 1))
    training_labels = to_categorical(testing_labels, NUM_CLASSES)

    # Load & Reshape Testing Images
    testing_images = np.genfromtxt(TESTING_IMAGES_PATH, dtype=np.single)
    testing_images = np.reshape(testing_images, (NUM_TESTING_IMAGES, IMG_HEIGHT, IMG_WIDTH, 1))


    return (testing_images, testing_labels)

def loadModel():

    json_file = open(MODEL_PATH, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(MODEL_WEIGHTS_PATH)

    return loaded_model

def testModel(model, x_test):

    y_pred = model.predict(x_test)

    out_file = open(OUTFILE_PATH , "w")
    out_file.write(str(NUM_TESTING_IMAGES) + "\n" )
    
    y_vec = np.zeros(NUM_CLASSES)

    for i in range(NUM_TESTING_IMAGES):

        for j in range(NUM_CLASSES):
            y_vec[j] = y_pred[i,j]

        y_type = np.argmax(y_vec)

        prob = y_vec[y_type]
        print('i=',i,'G-type=',y_type,'P',prob)

        out_file.write(str(y_type) + ' ' + str(y_vec[0]) + ' '+
                       str(y_vec[1]) + ' ' + str(y_vec[2]) + "\n" )

    out_file.close()
    
x_test, y_test = loadImages()
model = loadModel()
testModel(model, x_test)