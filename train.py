import keras
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from keras.models import Model
from keras.layers import Input
from keras.preprocessing import image
from keras.layers.convolutional import Conv2D
from keras.layers import Flatten, Reshape
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from sklearn.model_selection import train_test_split
from keras.layers import Dropout
from keras.layers.merge import concatenate
from keras.layers.pooling import GlobalMaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization

labels=np.loadtxt("../bird identification/CUB_200_2011/image_class_labels.txt",dtype=int)
labels=labels[:,1]
labels=labels-1
imagedata=np.load('image_data.npy')
labeldata=keras.utils.to_categorical(labels,200)
print(labeldata.shape)
X_train,X_test,y_train,y_test=train_test_split(imagedata,labeldata,test_size=1/6,stratify=labeldata)
print(X_train.shape)
print(y_train.shape)



model = Sequential()
model.add(Conv2D(input_shape=(200,200,3), filters=96, kernel_size=(3,3)))
model.add(Activation('relu'))
model.add(Conv2D(filters=96, kernel_size=(3,3), strides=2))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Conv2D(filters=192, kernel_size=(3,3)))
model.add(Activation('relu'))
model.add(Conv2D(filters=192, kernel_size=(3,3), strides=2))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(200, activation="softmax"))

sgd = keras.optimizers.SGD(lr=0.001, momentum=0.99, decay=1e-5, nesterov=True)
model.compile(loss=keras.losses.categorical_crossentropy,
      optimizer=sgd,
      metrics=['accuracy'])
history = model.fit(X_train, y_train,
      batch_size=128,
      epochs=50,
      verbose=1)
score=model.evaluate(X_test,y_test,verbose=0)
print('Test accuracy:',score[1])
plt.plot(history.history['acc'])
plt.show()
plt.plot(history.history['loss'])
plt.show()
