"""

This is for morphological classification of galaxies by CNN,
By Kenji Bekki, on 2020/2/14 for Nair & Abraham 2010
Refactored by Patrick Roe, on 2022/07/29

"""

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight

# Filenames for Data
TRAINING_IMAGES_PATH = "nam_images_train.dat"
TRAINING_LABELS_PATH = "nam_labels_train.dat"

# Image Data specifications
ORIG_IMG_HEIGHT = 50
ORIG_IMG_WIDTH = 50
UPSCALE_IMG_HEIGHT = 50
UPSCALE_IMG_WIDTH = 50
NUM_TRAINING_IMAGES = 11000
NUM_CLASSES = 2

NUM_TRAINING = 9500

# Training Parameters
NUM_EPOCHS = 100
BATCH_SIZE = 500
SHUFFLE = True

def upscaleImage(img):
    return cv2.resize(img, dsize=(UPSCALE_IMG_HEIGHT, UPSCALE_IMG_WIDTH), interpolation=cv2.INTER_CUBIC)

def loadImages():
    
    if Path('y_train.npy').exists() and Path('x_train.npy').exists():
        training_images = np.load(Path('x_train.npy'))
        training_labels = np.load(Path('y_train.npy'))
        return (training_images, training_labels)
    
    # Dictionary for converting to binary (Spiral & Non-Spiral) classification
    label_dict = {
        b"1": 0,
        b"2": 0,
        b"3": 1
    }
    
    # Converter to subtract 1 from all labels
    label_converter = lambda x: label_dict[x];               

    # Load & Reshape Training Labels
    training_labels = np.genfromtxt(TRAINING_LABELS_PATH, dtype=np.uint8, converters={0: label_converter})
    training_labels = np.reshape(training_labels, (NUM_TRAINING_IMAGES, 1))
    training_labels = to_categorical(training_labels, NUM_CLASSES)
    
    # Load & Reshape Training Images
    training_images = np.genfromtxt(TRAINING_IMAGES_PATH, dtype=np.single)
    training_images = np.reshape(training_images, (NUM_TRAINING_IMAGES, ORIG_IMG_HEIGHT, ORIG_IMG_WIDTH, 1))
    
    # Resize all images
    #new_training_images = np.zeros((NUM_TRAINING_IMAGES, UPSCALE_IMG_HEIGHT, UPSCALE_IMG_WIDTH))
    
    #for idx, img in enumerate(training_images):
    #    new_training_images[idx] = upscaleImage(img)
        
    #new_training_images = np.reshape(new_training_images, (NUM_TRAINING_IMAGES, UPSCALE_IMG_HEIGHT, UPSCALE_IMG_WIDTH, 1))

    np.save('y_train.npy', training_labels)
    np.save('x_train.npy', training_images)

    return (training_images, training_labels)

def trainModel(x_train, y_train):
    
    # Calculate the class weights of the data
    y_integers = np.argmax(y_train, axis=1)
    class_weights = class_weight.compute_class_weight(class_weight='balanced', 
                                                      classes=np.unique(y_integers), 
                                                      y=y_integers)
    class_weights = dict(enumerate(class_weights))

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(7, 7),
                    activation='relu',
                    input_shape=(UPSCALE_IMG_HEIGHT, UPSCALE_IMG_WIDTH, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(loss=categorical_crossentropy,
                optimizer=Adadelta(),
                metrics=['accuracy'])
    
    x_val = x_train[NUM_TRAINING:, :, :, :]
    y_val = y_train[NUM_TRAINING:, :]
    x_val = x_train[:NUM_TRAINING, :, :, :]
    y_val = y_train[:NUM_TRAINING, :]
    
    
    model.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, shuffle=SHUFFLE,
            verbose=1, validation_data=(x_val, y_val), class_weight=class_weights)

    return model

def saveModel(model):
    model_json = model.to_json()

    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # Serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

print("Loading images...")
x_train, y_train = loadImages()
print("Finished loading images.")
model = trainModel(x_train, y_train)
saveModel(model)