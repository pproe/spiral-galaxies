"""
This is for semantic segmentaion of galaxies by U-net
By Kenji Bekki, on 2020/2/25
Adapted by Patrick Roe, on 2022/11/19
Revised for two channel

"""


# import tensorflow.keras.backend.tensorflow_backend as KTF
# import tensorflow as tf
import os.path

import numpy as np
### Added 2018/3/30
# from tensorflow.keras.applications import imagenet_utils
# from tensorflow.keras.models import load_model
###
import tensorflow.keras.callbacks
from tensorflow.keras import backend as K
# from tensorflow import keras
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, Dropout, Input,
                                     MaxPooling2D, concatenate)
#### New addition
# from tensorflow.keras.layers import Activation, Flatten, Reshape
# from tensorflow.keras.layers.convolutional import Conv2D, Conv2D, MaxPooling2D, UpSampling2D
# from tensorflow.keras.layers.normalization import BatchNormalization
# from tensorflow.keras.utils import np_utils
from tensorflow.keras.models import Model, model_from_json

# from tensorflow.keras.utils import np_utils
# from tensorflow.keras.layers import Merge
# from tensorflow.keras.layers import Concatenate


### Total model number = (100*1) * nmodel

# iset=int(input('Input the total number of sets of models '))
# nmodel0=int(input('Input the total number of images per model '))
# nmodel=nmodel0*iset
# epochs=int(input('Input the number of epochs'))

iset = 5
epochs = 25
nmodel = 100
print("nmodel", nmodel)

### Original values
# batch_size = 128
# epochs = 12
num_classes = 2
batch_size = 1
# epochs = 500
nb_epoch = epochs
n_mesh = 64
# n_mesh=20
# nmodel=4000

# Channel number (new 2019/9/6)
nch = 1
# num class number
ncl = 1

img_rows, img_cols = n_mesh, n_mesh
n_mesh2 = n_mesh * n_mesh - 1
n_mesh3 = n_mesh * n_mesh


print(img_rows, img_cols, n_mesh2)


# input_shape = (img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, nch)


# This is for simlation data sets

#### Reading the density plot (1)
# 'tadaki_images.dat'
with open("set_2_orig.dat") as f:
    lines = f.readlines()

#### Reading the density plot (2)
# with open('2dft1.dat') as f:
#  lines0=f.readlines()

#### Reading the class file (one-hot vector: two class)
# 'tadaki_masks.dat'
with open("set_2_segmentation.dat") as f:
    lines1 = f.readlines()
# with open('id2.dat') as f:
#  lines2=f.readlines()

x_train = np.zeros((nmodel, n_mesh3, nch))
x_train0 = np.zeros((nmodel, n_mesh3, nch))
x_test = np.zeros((nmodel, n_mesh3, nch))

# y_train=np.zeros((nmodel,n_mesh3),dtype=np.int)
# y_test=np.zeros((nmodel,n_mesh3),dtype=np.int)
y_train = np.zeros((nmodel, n_mesh3, ncl))
y_test = np.zeros((nmodel, n_mesh3, ncl))
# y_train=np.zeros((nmodel,2))
# y_test=np.zeros((nmodel,2))


# For 2D density  map data
ibin = 0
jbin = -1
print(len(lines))
for num, j in enumerate(lines):
    jbin = jbin + 1
    tm = j.strip().split()
    x_train[ibin, jbin, 0] = float(tm[0])
    # x_train[ibin,jbin,1]=float(tm[0])
    # x_train[ibin,jbin,2]=float(tm[0])
    x_train0[ibin, jbin, 0] = float(tm[0])
    # x_train0[ibin,jbin,1]=float(tm[0])
    # x_train0[ibin,jbin,2]=float(tm[0])

    x_test[ibin, jbin, 0] = float(tm[0])
    # x_test[ibin,jbin,1]=float(tm[0])
    # x_test[ibin,jbin,2]=float(tm[0])
    # print('ibin,jbin',ibin,jbin)
    if jbin == n_mesh2:
        ibin += 1
        jbin = -1

# For 2D V_los  map data
# ibin=0
# jbin=-1
# for num,j in enumerate(lines0):
#  jbin=jbin+1
#  tm=j.strip().split()
#  x_train[ibin,jbin,1]=float(tm[0])
#  x_test[ibin,jbin,1]=float(tm[0])
#  if jbin == n_mesh2:
#    ibin+=1
#    jbin=-1


# For class allocation for each mesh (segmentation)
ibin = 0
jbin = -1
for num, j in enumerate(lines1):
    jbin = jbin + 1
    tm = j.strip().split()
    #  y_train[ibin,jbin]=int(tm[0])
    #  y_test[ibin,jbin]=int(tm[0])
    #  y_train[ibin,jbin]=int(tm[0])-1
    #  y_test[ibin,jbin]=int(tm[0])-1
    # For classification
    #  y_train[ibin,jbin,0]=int(tm[0])
    #  y_test[ibin,jbin,0]=int(tm[0])
    # For regression
    y_train[ibin, jbin, 0] = float(tm[0])
    y_test[ibin, jbin, 0] = float(tm[0])
    if jbin == n_mesh2:
        ibin += 1
        jbin = -1

# For class allocation for each mesh (segmentation)
# ibin=0
# jbin=-1
# for num,j in enumerate(lines2):
#  jbin=jbin+1
#  tm=j.strip().split()
#  y_train[ibin,jbin,1]=int(tm[0])
#  y_test[ibin,jbin,1]=int(tm[0])
#  if jbin == n_mesh2:
#    ibin+=1
#    jbin=-1


x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, nch)
x_train0 = x_train0.reshape(x_train0.shape[0], img_rows, img_cols, nch)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, nch)

y_train = y_train.reshape(y_train.shape[0], img_rows, img_cols, ncl)
y_test = y_test.reshape(y_test.shape[0], img_rows, img_cols, ncl)
# y_train = tensorflow.keras.utils.np_utils.to_categorical(y_train, num_classes)
# y_test =  tensorflow.keras.utils.np_utils.to_categorical(y_test, num_classes)
# y_train = tensorflow.keras.utils.np_utils.to_categorical(y_train, ncl)
# y_test =  tensorflow.keras.utils.np_utils.to_categorical(y_test, ncl)


# print('Galaxy type',y_train[:5])

# input_shape=(img_rows,img_cols,nch)
# nrc=img_rows*img_cols
# print('nch',nch)

print("input_shape", input_shape)

inputs = Input(shape=input_shape)

# Add augmentation
seed = 24
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_data_gen_args = dict(
    rotation_range=90,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.5,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="reflect",
)

mask_data_gen_args = dict(
    rotation_range=90,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.5,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="reflect",
    preprocessing_function=lambda x: np.where(x > 0, 1, 0).astype(x.dtype),
)

seed = 25
validation_seed = 24

# Training data generator
image_data_generator = ImageDataGenerator(**img_data_gen_args)
image_data_generator.fit(x_train, augment=True, seed=seed)
image_generator = image_data_generator.flow(x_train, batch_size=batch_size, seed=seed)

mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
mask_data_generator.fit(y_train, augment=True, seed=seed)
mask_generator = mask_data_generator.flow(y_train, batch_size=batch_size, seed=seed)

# Validation data generator
val_image_data_generator = ImageDataGenerator(**img_data_gen_args)
val_image_data_generator.fit(x_test, augment=True, seed=validation_seed)
val_image_generator = image_data_generator.flow(
    x_test, batch_size=batch_size, seed=validation_seed
)

val_mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
val_mask_data_generator.fit(y_test, augment=True, seed=validation_seed)
val_mask_generator = mask_data_generator.flow(
    y_test, batch_size=batch_size, seed=validation_seed
)


def image_mask_generator(image, mask):
    train = zip(image, mask)
    for img, mask in train:
        yield (img, mask)


val_generator = image_mask_generator(val_image_generator, val_mask_generator)
train_generator = image_mask_generator(image_generator, mask_generator)

# Contraction path
c1 = Conv2D(
    16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
)(inputs)
c1 = Dropout(0.1)(c1)
c1 = Conv2D(
    16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
)(c1)
p1 = MaxPooling2D((2, 2))(c1)

c2 = Conv2D(
    32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
)(p1)
c2 = Dropout(0.1)(c2)
c2 = Conv2D(
    32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
)(c2)
p2 = MaxPooling2D((2, 2))(c2)

c3 = Conv2D(
    64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
)(p2)
c3 = Dropout(0.2)(c3)
c3 = Conv2D(
    64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
)(c3)
p3 = MaxPooling2D((2, 2))(c3)

c4 = Conv2D(
    128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
)(p3)
c4 = Dropout(0.2)(c4)
c4 = Conv2D(
    128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
)(c4)
p4 = MaxPooling2D(pool_size=(2, 2))(c4)

c5 = Conv2D(
    256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
)(p4)
c5 = Dropout(0.3)(c5)
c5 = Conv2D(
    256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
)(c5)

# Expansive path
u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(
    128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
)(u6)
c6 = Dropout(0.2)(c6)
c6 = Conv2D(
    128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
)(c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(
    64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
)(u7)
c7 = Dropout(0.2)(c7)
c7 = Conv2D(
    64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
)(c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(
    32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
)(u8)
c8 = Dropout(0.1)(c8)
c8 = Conv2D(
    32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
)(c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(
    16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
)(u9)
c9 = Dropout(0.1)(c9)
c9 = Conv2D(
    16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
)(c9)

outputs = Conv2D(1, (1, 1), activation="sigmoid")(c9)

z = outputs.shape
print(z)

model = Model(inputs, outputs)


model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# model.compile(loss='mean_squared_error',
#              optimizer=tensorflow.keras.optimizers.Adadelta(),
#              metrics=['accuracy'])
# model.compile(loss="categorical_crossentropy", optimizer='adadelta',
#              metrics=["accuracy"])


steps_per_epoch = 12 * (len(x_train)) // batch_size
model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_steps=steps_per_epoch,
    epochs=epochs,
    verbose=1,
    validation_data=val_generator,
    shuffle=True,
)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test score:", score[0])
print("Test accuracy:", score[1])


print("save the architecture of a model")

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
