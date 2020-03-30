#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 13:54:42 2020

@author: swilson
"""
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.keras.preprocessing import image
from PIL import Image
import numpy as np
import os


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']= '-1'

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
    
    
def run_classify_image(img):
    """
    * img: converted to numpy array with shape (1, 265, 265,3)
    * returns classification prediction 
      positive if predictions[0][0] > predictions[0][1] 
               else negative
    """   
    f = gfile.FastGFile("tf-models/tf_model.pb", 'rb')
    graph_def = tf.GraphDef()
   # Parses a serialized binary message into the current message.
    graph_def.ParseFromString(f.read())
    f.close()
    
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)        
    tensor_names = []
    for op in graph.get_operations():
        tensor_names.append(op.name) 
    output_name = [tensor for tensor in tensor_names 
                   if ('Softmax' in tensor)][0] + ':0'
    input_name = [tensor for tensor in tensor_names 
                  if ('input' in tensor)][0] + ':0'

    sess = tf.Graph()
    with sess.as_default() as graph:
        tf.import_graph_def(graph_def)
        softmax_tensor = sess.get_tensor_by_name(output_name)  

    with tf.Session(graph=graph) as sess:
        predictions = sess.run(softmax_tensor, {input_name: img}) 
         
    return predictions  



############################ 
#   Testing
############################ 
    
###--- (1) convert .png to numpay array
def img_to_array(raw_img):
    img_width, img_height = 256, 256
    img = image.load_img(raw_img, target_size=(img_width, img_height))
    img = image.img_to_array(img)/255.
    img = np.expand_dims(img, axis=0)
    return img

img1 = img_to_array('./highway_primary/TEST/test_primary/16_34579_22088.png')
img2 = img_to_array('./highway_primary/TEST/test_not_primary/19_256846_198330.png')
    
###--- (2) Testing
def classifier(img):
    positive = run_classify_image(img)[0][0]
    negative = run_classify_image(img)[0][1]
    if positive < negative:
        dic = True
    else:
        dic = False
    return dic


classifier(img1)
classifier(img2)




