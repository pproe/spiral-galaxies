"""

This is for morphological classification of galaxies by CNN,
By Kenji Bekki, on 2017/11/15
Revised on 2020/2/14 (Nair & Abraham 2010)
For test only.

"""


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

from sklearn.metrics import confusion_matrix

### Original values
#batch_size = 128
#num_classes = 10
#epochs = 12
#batch_size = 200

num_classes = 3
num_classes0 = 2

#epochs = 1
#nb_epoch=epochs

n_mesh=50

img_rows, img_cols = n_mesh, n_mesh
n_mesh2=n_mesh*n_mesh-1
n_mesh3=n_mesh*n_mesh

#input_shape = (img_rows, img_cols, 1)

# For output the galaxy classification results
f1=open('test27d.out','w')

# This is for simlation data sets
with open('2dfv.dat') as f:
  lines=f.readlines()

nmodel=542

x_train=np.zeros((nmodel,n_mesh3))
x_test=np.zeros((nmodel,n_mesh3))
y_train=np.zeros(nmodel,dtype=np.int)
y_test=np.zeros(nmodel,dtype=np.int)



#y_test=np.zeros(nmodel)
#print(y_train)

# For 2D density map data
ibin=0
jbin=-1
for num,j in enumerate(lines):
  jbin=jbin+1
  tm=j.strip().split()
  x_train[ibin,jbin]=float(tm[0])
  x_test[ibin,jbin]=float(tm[0])
#  print('ibin,jbin',ibin,jbin)
  if jbin == n_mesh2:
    ibin+=1
    jbin=-1

ntest=ibin
print('ntest',ntest)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
#y_test =  keras.utils.np_utils.to_categorical(y_test, num_classes)


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

#y_vec=np.zeros(3)
#y_vec=np.zeros(num_classes)
y_vec=np.zeros(num_classes)
#print(y_vec)

y_pred=loaded_model.predict(x_test)
print(y_pred[:ntest])

f1.write( str(ntest) + "\n" )

result = confusion_matrix(y_test, y_pred , normalize='pred')

for i in range(ntest):
#  for j in range(num_classes0):
  for j in range(num_classes):
    y_vec[j]=y_pred[i,j]
#  print(y_vec)
#    print(j)
  y_type=np.argmax(y_vec)
#  y_type=y_type+1
  prob=y_vec[y_type]
  print('i=',i,'G-type=',y_type,'P',prob)
#  Original  type-1 is output
  f1.write( str(y_type) + ' ' + str(y_vec[0]) + ' '+
  str(y_vec[1]) + ' ' + str(y_vec[2]) + "\n" )
#  f1.write( str(y_type) +
#   "\n" )


#loaded_model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.Adadelta(),
#              metrics=['accuracy'])



