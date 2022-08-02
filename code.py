# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 20:14:14 2020

@author: peter
"""
# In[*]
# TensorFlow and tf.keras
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras import Model

from tensorflow.keras.layers import  Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import orthogonal, he_normal
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import Conv2D,  Input,  Reshape
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
#%%

input_height=256
input_width=256
nClasses=3
nChannels=3
inputs = Input(shape=(input_height, input_width, nChannels))

conv1 = Conv2D(64, (3, 3),dilation_rate=[1,2], activation='relu', padding='same', kernel_initializer=orthogonal())(inputs)
conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(pool1)
conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

dcon1 = Conv2D(128, (3, 3), dilation_rate=2, activation='relu', padding='same', kernel_initializer=orthogonal())(pool2)
dcon2 = Conv2D(128, (3, 3), dilation_rate=3, activation='relu', padding='same', kernel_initializer=orthogonal())(dcon1)
dcon3 = Conv2D(128, (3, 3), dilation_rate=5, activation='relu', padding='same', kernel_initializer=orthogonal())(dcon2)
pool4 = MaxPooling2D(pool_size=(2, 2))(dcon3)


dcon4 = Conv2D(256, (3, 3), dilation_rate=2, activation='relu', padding='same', kernel_initializer=orthogonal())(pool4)
dcon5 = Conv2D(256, (3, 3), dilation_rate=3, activation='relu', padding='same', kernel_initializer=orthogonal())(dcon4)
dcon6 = Conv2D(256, (3, 3), dilation_rate=5, activation='relu', padding='same', kernel_initializer=orthogonal())(dcon5)


up6 = UpSampling2D(size=(2, 2))(dcon6)
up6 = concatenate([up6, dcon3], axis=-1)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(up6)

up7 = UpSampling2D(size=(2, 2))(conv6)
up7 = concatenate([up7, conv2], axis=-1)
conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(up7)

up8 = UpSampling2D(size=(2, 2))(conv7)
up8 = concatenate([up8, conv1], axis=-1)
conv8 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(up8)



up10 = concatenate([conv8, inputs], axis=-1)
conv10 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(up10)

conv11 = Conv2D(nClasses, (1, 1), padding='same', activation='softmax',
                kernel_initializer=he_normal(), kernel_regularizer=l2(0.005))(conv10)
conv11 = Reshape((-1, nClasses))(conv11)


model = Model(inputs,conv11)

model.compile(optimizer = Adam(lr=1e-4), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.summary()


