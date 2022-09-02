"""

This is for morphological classification of galaxies by CNN,
By Kenji Bekki, on 2020/2/14 for Nair & Abraham 2010

"""

from re import M

from sklearn.utils import shuffle
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from keras.models import model_from_json
import keras.callbacks
import numpy as np
#import keras.backend.tensorflow_backend as KTF
#import tensorflow as tf
import os.path


### Total model number = (nmodle0) * nmodel

#iset=int(input('Input the total number of sets of models '))
#nmodel0=int(input('Input the total number of images per model'))
#nmodel=nmodel0*iset
nmodel=2458
print('nmodel',nmodel)

### Original values
#batch_size = 128
#num_classes = 10
#epochs = 12
batch_size = 100 # Number of samples per gradient update
#num_classes = 5
num_classes = 3 # Different types of galaxies
epochs = 30      
nb_epoch=epochs
n_mesh=50       # Size of the images (width & height)
#nmodel=1000
print('nmodel',nmodel)
print('num_classes',num_classes)

img_rows, img_cols = n_mesh, n_mesh 
n_mesh2=n_mesh*n_mesh-1             #2499
n_mesh3=n_mesh*n_mesh               #2500


print(img_rows, img_cols, n_mesh2)
#stop



#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1) # (50, 50, 1)

#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255
#print('x_train shape:', x_train.shape)
#print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')
#print(y_test.shape[0], 'y.test samples')
#print(str(y_test[0]))
#print(str(y_test[1]))
#print(str(y_test[2]))

#y_train = y_train.astype('int32')
#y_test = y_test.astype('int32')
#y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
#y_test =  keras.utils.np_utils.to_categorical(y_test, num_classes)

# This is for simlation data sets

with open('2dft.dat') as f:
  lines=f.readlines()
with open('2dftn.dat') as f:
  lines1=f.readlines()


x_train=np.zeros((nmodel,n_mesh3))    # Empty array for training images
x_test=np.zeros((nmodel,n_mesh3))     # Empty array for testing images
y_train=np.zeros(nmodel,dtype=np.int) # Empty array for training labels
y_test=np.zeros(nmodel,dtype=np.int)  # Empty array for testing labels
#y_test=np.zeros(nmodel)
#print(y_train)

# For 2D density map data
ibin=0
jbin=-1
for num,j in enumerate(lines):        # For each line in training images data (1 line per pixel)
  jbin=jbin+1                         # Iterate jbin
  tm=j.strip().split()                # Strip whitespace and split on whitespace
  x_train[ibin,jbin]=float(tm[0])     # Assign exact same data to testing image array
  x_test[ibin,jbin]=float(tm[0])      # Assign exact same data to testing image array
#  print('ibin,jbin',ibin,jbin)
  if jbin == n_mesh2:                 # If jbin == 2499 reached end of current image data
    ibin+=1                           # Iterate image number
    jbin=-1                           # Reset jbin to -1

# For morphological map
ibin=0
for num,j in enumerate(lines1):
  tm=j.strip().split()
  y_train[ibin]=int(tm[0])-1
  y_test[ibin]=int(tm[0])-1
#  print('ibin, (Morpholigcl type)',ibin,y_train[ibin])
  ibin+=1




x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test =  keras.utils.np_utils.to_categorical(y_test, num_classes)

print('Galaxy type',y_train[:5])

#stop

print(x_train.shape)
print(y_train.shape)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True,
          verbose=1, validation_data=(x_test, y_test)) #validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


print('save the architecture of a model')

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")





