"""

This is for morphological classification of galaxies by CNN,
By Kenji Bekki, on 2017/11/15
Revised on 2020/2/14 (Nair & Abraham 2010)
Refactored by Patrick Roe, on 2022/07/29
For test only.

"""

import numpy as np
import cv2
import os

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import model_from_json

# Filenames for Data
TESTING_IMAGES_PATH = "..\FileManipulation\Tadaki_images.dat"
TESTING_LABELS_PATH = "..\FileManipulation\Tadaki_labels.dat"
OUTFILE_PATH = "test.out"

# Filenames for Model
MODEL_PATH = "model.json"
MODEL_WEIGHTS_PATH = ".\checkpoints\ 39.h5"

# Image Data specifications
ORIG_IMG_HEIGHT = 64
ORIG_IMG_WIDTH = 64
UPSCALE_IMG_HEIGHT = 64
UPSCALE_IMG_WIDTH = 64
NUM_TESTING_IMAGES = 10000
NUM_CLASSES = 2


def upscaleImage(img):
    return cv2.resize(
        img,
        dsize=(UPSCALE_IMG_HEIGHT, UPSCALE_IMG_WIDTH),
        interpolation=cv2.INTER_CUBIC,
    )


def loadData():
    # Dictionary for converting to binary (Spiral & Non-Spiral) classification
    """label_dict = {
        b"1": 0,
        b"2": 0,
        b"3": 1
    }

    # Converter to subtract 1 from all labels
    label_converter = lambda x: label_dict[x];"""

    # Load & Reshape Testing Labels
    testing_labels = np.genfromtxt(
        TESTING_LABELS_PATH, dtype=np.uint8, max_rows=NUM_TESTING_IMAGES
    )  # , converters={0: label_converter})
    testing_labels = np.reshape(testing_labels, (NUM_TESTING_IMAGES, 1))
    # testing_labels = to_categorical(testing_labels, NUM_CLASSES)

    # Load & Reshape Testing Images
    testing_images = np.genfromtxt(
        TESTING_IMAGES_PATH,
        dtype=np.single,
        max_rows=NUM_TESTING_IMAGES * ORIG_IMG_WIDTH * ORIG_IMG_WIDTH,
    )
    testing_images = np.reshape(
        testing_images, (NUM_TESTING_IMAGES, ORIG_IMG_HEIGHT, ORIG_IMG_WIDTH, 1)
    )

    # Resize all images
    new_testing_images = np.zeros(
        (NUM_TESTING_IMAGES, UPSCALE_IMG_HEIGHT, UPSCALE_IMG_WIDTH)
    )

    for idx, img in enumerate(testing_images):
        new_testing_images[idx] = upscaleImage(img)

    new_testing_images = np.reshape(
        new_testing_images,
        (NUM_TESTING_IMAGES, UPSCALE_IMG_HEIGHT, UPSCALE_IMG_WIDTH, 1),
    )

    return (new_testing_images, testing_labels)


def loadModel():
    json_file = open(MODEL_PATH, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(MODEL_WEIGHTS_PATH)

    return loaded_model


def testModel(model, x_test):
    y_pred = model.predict(x_test)

    out_file = open(OUTFILE_PATH, "w")
    out_file.write(str(NUM_TESTING_IMAGES) + "\n")

    for i in range(NUM_TESTING_IMAGES):
        y_type = round(y_pred[i, 0])

        print("i=", i, "G-type=", y_type)

        out_file.write(str(y_type) + "\n")

    out_file.close()


x_test, y_test = loadData()
model = loadModel()
testModel(model, x_test)
