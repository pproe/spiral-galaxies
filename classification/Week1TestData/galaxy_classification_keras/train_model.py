"""

This is for morphological classification of galaxies by CNN,
By Kenji Bekki, on 2020/2/14 for Nair & Abraham 2010
Refactored by Patrick Roe, on 2022/07/29

"""

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical

# Filenames for Data
TRAINING_IMAGES_PATH = "2dft.dat"
TRAINING_LABELS_PATH = "2dftn.dat"

# Image Data specifications
IMG_HEIGHT = 50
IMG_WIDTH = 50
NUM_TRAINING_IMAGES = 2458
NUM_CLASSES = 3

# Training Parameters
NUM_EPOCHS = 200
BATCH_SIZE = 50
SHUFFLE = True

def loadImages():

    # Converter to subtract 1 from all labels
    label_converter = lambda x: int(x) - 1;               

    # Load & Reshape Training Labels
    training_labels = np.genfromtxt(TRAINING_LABELS_PATH, dtype=np.uint8, converters={0: label_converter})
    training_labels = np.reshape(training_labels, (NUM_TRAINING_IMAGES, 1))
    training_labels = to_categorical(training_labels, NUM_CLASSES)

    # Load & Reshape Training Images
    training_images = np.genfromtxt(TRAINING_IMAGES_PATH, dtype=np.single)
    training_images = np.reshape(training_images, (NUM_TRAINING_IMAGES, IMG_HEIGHT, IMG_WIDTH, 1))

    return (training_images, training_labels)

def trainModel(x_train, y_train):

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(loss=categorical_crossentropy,
                optimizer=Adadelta(),
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, shuffle=SHUFFLE,
            verbose=1, validation_data=(x_train, y_train))

    return model

def saveModel(model):
    model_json = model.to_json()

    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # Serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

x_train, y_train = loadImages()
model = trainModel(x_train, y_train)
saveModel(model)