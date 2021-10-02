"""
......author @Punith kumar
"""

# importing all the libraries required

import pandas as pd

import tensorflow as tf
import keras

from keras.applications.vgg import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten,Dense

from glob import glob

import matplotlib.pyplot as plt

# Data Augumentation

train_path='/content/chest_xray/train'
test_path='/content/chest_xray/test'
val_path='/content/chest_xray/val'
folders = glob('/content/chest_xray/train/*')

datagen =ImageDataGenerator(rescale = 1./255)

train_data= datagen.flow_from_directory('/content/chest_xray/train',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

test_data = datagen.flow_from_directory('/content/chest_xray/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')
val_data = datagen.flow_from_directory('/content/chest_xray/val',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')
                                            
#Building the Model

vgg= VGG16(input_shape=(224,224,3),include_top='False',weights='imagenet')

for layer in vgg.layers:
  layer.trainable='False'
  
x=Flatten()(vgg.output)
output=Dense(len(folders),activation="softmax")(x)

model=keras.Model(inputs=vgg.input,outputs=output)
model.summary()

#compiling the model
  model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy') 

#Training the model
  model.fit_generator(train_data,epochs=5,validation_data=val_data)

#Visualising Loss of the model 
  loss=pd.DataFrame(model.history.history).plot()
  
#Fine tuning of entire model with a small learning rate

model.trainable=True

model.compile(optimizer=keras.optimizers.Adam(1e-5),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=10,validation_split=0.2)

loss=pd.DataFrame(model.history.history).plot()

# saving the model
  model.save('x-ray_model.h5')
