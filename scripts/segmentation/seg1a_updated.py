"""
This is for semantic segmentaion of galaxies by U-net
By Kenji Bekki, on 2020/2/25
Adapted by Patrick Roe, on 2022/11/19

Revised for two channel

"""


# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras import backend as K
# from keras.utils import np_utils
from tensorflow.keras.models import model_from_json

### Added 2018/3/30
# from keras.applications import imagenet_utils
# from keras.models import load_model
###
# import keras.callbacks
import numpy as np

# import keras.backend.tensorflow_backend as KTF
# import tensorflow as tf
import os.path

#### New addition
# from keras.layers import Input
# from keras.layers.core import Activation, Flatten, Reshape
# from keras.layers.convolutional import Convolution2D, Conv2D, MaxPooling2D, UpSampling2D
# from keras.layers.normalization import BatchNormalization
# from keras.models import Model
# from keras.utils import np_utils
# from keras.layers import Merge
# from keras.layers import Concatenate


### Total model number = (100*1) * nmodel

# iset=int(input('Input the total number of sets of models '))
# nmodel0=int(input('Input the total number of images per model '))
# nmodel=nmodel0*iset
# epochs=int(input('Input the number of epochs'))

iset = 5
epochs = 10
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

# For total

ntest = nmodel * n_mesh3


print("ntest", ntest)
print("img_rows,cols,mesh2", img_rows, img_cols, n_mesh2)


# input_shape = (img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, nch)


# This is for simlation data sets

#### Reading the density plot (1)
with open("set_2_orig.dat") as f:
    lines = f.readlines()

#### Reading the density plot (2)
# with open('2dft1.dat') as f:
#  lines0=f.readlines()

#### For output the segmentation label
f1 = open("set_2_pred.dat", "w")

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
for num, j in enumerate(lines):
    jbin = jbin + 1
    tm = j.strip().split()
    x_train[ibin, jbin, 0] = float(tm[0])
    # x_train[ibin,jbin,1]=float(tm[0])
    # x_train[ibin,jbin,2]=float(tm[0])

    x_test[ibin, jbin, 0] = float(tm[0])
    # x_test[ibin,jbin,1]=float(tm[0])
    # x_test[ibin,jbin,2]=float(tm[0])

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


x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, nch)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, nch)

# y_train = y_train.reshape(y_train.shape[0], img_rows, img_cols, ncl)
# y_test = y_test.reshape(y_test.shape[0], img_rows, img_cols, ncl)
# y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
# y_test =  keras.utils.np_utils.to_categorical(y_test, num_classes)
# y_train = keras.utils.np_utils.to_categorical(y_train, ncl)
# y_test =  keras.utils.np_utils.to_categorical(y_test, ncl)


###
# load json and create model
json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

y_pred = loaded_model.predict(x_test)
yps = y_pred.shape
# print('y_pred',y_pred[:ntest])
print("y_pred-shape", yps)

# f1.write( str(ntest) + "\n" )

it = 0
for i in range(nmodel):
    for j in range(img_rows):
        for k in range(img_cols):
            for k1 in range(ncl):
                yv0 = y_pred[i, j, k, k1]
                #     print('i=',i,'j',j,'k',k,'yv0',yv0)
                f1.write(str(yv0) + "\n")
                it = it + 1

f1.close()
print("total", it)
