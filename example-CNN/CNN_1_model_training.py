#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:40:40 2020

@author: swilson
"""

import os
import numpy as np
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import keras
from tensorflow.python.platform import gfile


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']= '0,1'

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


#####===== (STEP 1) Data Preparation - Import datasets, set image sie
#  - train_data_dir: directory path where training data sets are
#  - test_data_dir:  directory path where testing data sets are
#  - train_pos_dir: directory name where positive data for training
#  - train_neg_dir: directory name where negative data for training
#  - test_pos_dir: directory name where positive data for testing
#  - test_neg_dir: directory name where negative data for testing
#
#  - epochs: an epoch is one learning cycle where the learner sees the whole training data set
#  - batch_size: the number of images that will be propagated through the network 
#  - trained_model: HDF5(.h5) file name that will save best model's parameters



train_data_dir = './highway_primary/TRAIN/'
validation_data_dir = './highway_primary/TEST/'

train_pos_dir = 'train_primary'
train_neg_dir = 'train_not_primary'

test_pos_dir = 'test_primary'
test_neg_dir = 'test_not_primary'

epochs = 50
batch_size = 16
trained_model = 'keras.h5'


num_train_pos = len([file for file in os.listdir(os.path.join(train_data_dir, train_pos_dir))])
num_train_others = len([file for file in os.listdir(os.path.join(train_data_dir, train_neg_dir))])
num_test_pos = len([file for file in os.listdir(os.path.join(validation_data_dir, test_pos_dir))])
num_test_neg = len([file for file in os.listdir(os.path.join(validation_data_dir, test_neg_dir))])

nb_train_samples = num_train_pos + num_train_others
nb_validation_samples = num_test_pos + num_test_neg

epochs = epochs
batch_size = batch_size


    ###---  configure the shape of dataset
img_width, img_height = 256, 256
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
    
    
#####===== (STEP 2) Model: Convolutional Nueral Network
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))  #--32
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))               
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', 
              optimizer='rmsprop',metrics=['accuracy'])


#####===== (STEP 3) Learning the model (Save the best performing models as .h5 file)
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_width, img_height), 
                                                    batch_size=batch_size, class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(validation_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

# Save the model according to the conditions  
checkpoint = ModelCheckpoint(trained_model, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

# Model fitting and Validation
model.fit_generator(train_generator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size,
                    callbacks = [checkpoint, early])


#####===== (STEP 4) Save "tf_model.pb" in ./tf-models directory
K.set_learning_phase(0)

model = load_model(trained_model)

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, "tf-models", "tf_model.pb", as_text=False)










