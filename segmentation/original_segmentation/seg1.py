"""
This is for semantic segmentaion of galaxies by U-net
By Kenji Bekki, on 2020/2/25

Revised for two channel

"""


#from tensorflow import keras
#from tensorflow.keras.datasets import mnist
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Flatten, MaxPooling2D, Conv2D, UpSampling2D, Concatenate
from tensorflow.keras import backend as K
#from tensorflow.keras.utils import np_utils
from tensorflow.keras.models import model_from_json
### Added 2018/3/30
#from tensorflow.keras.applications import imagenet_utils
#from tensorflow.keras.models import load_model
###
import tensorflow.keras.callbacks
import numpy as np
#import tensorflow.keras.backend.tensorflow_backend as KTF
#import tensorflow as tf
import os.path
#### New addition
from tensorflow.keras.layers import Input
#from tensorflow.keras.layers import Activation, Flatten, Reshape
#from tensorflow.keras.layers.convolutional import Conv2D, Conv2D, MaxPooling2D, UpSampling2D
#from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.models import Model
#from tensorflow.keras.utils import np_utils
#from tensorflow.keras.layers import Merge
#from tensorflow.keras.layers import Concatenate


### Total model number = (100*1) * nmodel

#iset=int(input('Input the total number of sets of models '))
#nmodel0=int(input('Input the total number of images per model '))
#nmodel=nmodel0*iset
#epochs=int(input('Input the number of epochs'))

iset=5
epochs=10
nmodel=2500
print('nmodel',nmodel)

### Original values
#batch_size = 128
#epochs = 12
num_classes = 2
batch_size = 16
#epochs = 500
nb_epoch=epochs
n_mesh=64
#n_mesh=20
#nmodel=4000

#Channel number (new 2019/9/6)
nch=3
#num class number
ncl=1

img_rows, img_cols = n_mesh, n_mesh
n_mesh2=n_mesh*n_mesh-1
n_mesh3=n_mesh*n_mesh


print(img_rows, img_cols, n_mesh2)


#input_shape = (img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, nch)


# This is for simlation data sets

#### Reading the density plot (1)
with open('2df.dat') as f:
  lines=f.readlines()

#### Reading the density plot (2)
#with open('2dft1.dat') as f:
#  lines0=f.readlines()

#### Reading the class file (one-hot vector: two class)
with open('id1.dat') as f:
  lines1=f.readlines()
with open('id2.dat') as f:
  lines2=f.readlines()

x_train=np.zeros((nmodel,n_mesh3,nch))
x_train0=np.zeros((nmodel,n_mesh3,nch))
x_test=np.zeros((nmodel,n_mesh3,nch))

#y_train=np.zeros((nmodel,n_mesh3),dtype=np.int)
#y_test=np.zeros((nmodel,n_mesh3),dtype=np.int)
y_train=np.zeros((nmodel,n_mesh3,ncl))
y_test=np.zeros((nmodel,n_mesh3,ncl))
#y_train=np.zeros((nmodel,2))
#y_test=np.zeros((nmodel,2))


# For 2D density  map data
ibin=0
jbin=-1
print(len(lines))
for num,j in enumerate(lines):
  jbin=jbin+1
  tm=j.strip().split()
  x_train[ibin,jbin,0]=float(tm[0])
  x_train[ibin,jbin,1]=float(tm[0])
  x_train[ibin,jbin,2]=float(tm[0])
  x_train0[ibin,jbin,0]=float(tm[0])
  x_train0[ibin,jbin,1]=float(tm[0])
  x_train0[ibin,jbin,2]=float(tm[0])

  x_test[ibin,jbin,0]=float(tm[0])
  x_test[ibin,jbin,1]=float(tm[0])
  x_test[ibin,jbin,2]=float(tm[0])
  #print('ibin,jbin',ibin,jbin)
  if jbin == n_mesh2:
    ibin+=1
    jbin=-1

# For 2D V_los  map data
#ibin=0
#jbin=-1
#for num,j in enumerate(lines0):
#  jbin=jbin+1
#  tm=j.strip().split()
#  x_train[ibin,jbin,1]=float(tm[0])
#  x_test[ibin,jbin,1]=float(tm[0])
#  if jbin == n_mesh2:
#    ibin+=1
#    jbin=-1


# For class allocation for each mesh (segmentation)
ibin=0
jbin=-1
for num,j in enumerate(lines1):
  jbin=jbin+1
  tm=j.strip().split()
#  y_train[ibin,jbin]=int(tm[0])
#  y_test[ibin,jbin]=int(tm[0])
#  y_train[ibin,jbin]=int(tm[0])-1
#  y_test[ibin,jbin]=int(tm[0])-1
# For classification
#  y_train[ibin,jbin,0]=int(tm[0])
#  y_test[ibin,jbin,0]=int(tm[0])
# For regression
  y_train[ibin,jbin,0]=float(tm[0])
  y_test[ibin,jbin,0]=float(tm[0])
  if jbin == n_mesh2:
    ibin+=1
    jbin=-1

# For class allocation for each mesh (segmentation)
#ibin=0
#jbin=-1
#for num,j in enumerate(lines2):
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
#y_train = tensorflow.keras.utils.np_utils.to_categorical(y_train, num_classes)
#y_test =  tensorflow.keras.utils.np_utils.to_categorical(y_test, num_classes)
#y_train = tensorflow.keras.utils.np_utils.to_categorical(y_train, ncl)
#y_test =  tensorflow.keras.utils.np_utils.to_categorical(y_test, ncl)


#print('Galaxy type',y_train[:5])






#input_shape=(img_rows,img_cols,nch)
#nrc=img_rows*img_cols
#print('nch',nch)

inputs = Input(shape=input_shape)

conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2), padding="same")(conv1)

conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2), padding="same")(conv2)

conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2), padding="same")(conv3)

conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2), padding="same")(conv4)

conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
pool5 = MaxPooling2D(pool_size=(2, 2), padding="same")(conv5)

conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool5)
conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv6)

#up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv5], mode='concat', concat_axis=3)
up_conv6=UpSampling2D(size=(2, 2))(conv6)
up7 = Concatenate(axis=3)([up_conv6, conv5])
conv7 = Conv2D(512, (3, 3), activation='relu', padding='same')(up7)
conv7 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv7)

#up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv4], mode='concat', concat_axis=3)
up_conv7=UpSampling2D(size=(2, 2))(conv7)
up8 = Concatenate(axis=3)([up_conv7, conv4])
conv8 = Conv2D(256, (3, 3), activation='relu', padding='same')(up8)
conv8 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv8)

#up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv3], mode='concat', concat_axis=3)
up_conv8=UpSampling2D(size=(2, 2))(conv8)
up9 = Concatenate(axis=3)([up_conv8, conv3])
conv9 = Conv2D(128, (3, 3), activation='relu', padding='same')(up9)
conv9 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv9)

#up10 = merge([UpSampling2D(size=(2, 2))(conv9), conv2], mode='concat', concat_axis=3)
up_conv9=UpSampling2D(size=(2, 2))(conv9)
up10 = Concatenate(axis=3)([up_conv9, conv2])
conv10 = Conv2D(64, (3, 3), activation='relu', padding='same')(up10)
conv10 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv10)

#up11 = merge([UpSampling2D(size=(2, 2))(conv10), conv1], mode='concat', concat_axis=3)
up_conv10=UpSampling2D(size=(2, 2))(conv10)
up11 = Concatenate(axis=3)([up_conv10, conv1])
conv11 = Conv2D(32, (3, 3), activation='relu', padding='same')(up11)
conv11 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv11)

conv12 = Conv2D(1, 1, 1, activation='sigmoid')(conv11)

z=conv12.shape
print(z)

model = Model(inputs, conv12)




model.compile(loss='mean_squared_error',
              optimizer=tensorflow.keras.optimizers.Adadelta(),
              metrics=['accuracy'])
#model.compile(loss="categorical_crossentropy", optimizer='adadelta', 
#              metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


print('save the architecture of a model')

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")




