#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:42:22 2020

@author: swilson
"""
import json
import boto3
import numpy as np
import base64
import io
import cv2

from PIL import Image
import matplotlib.pyplot as plt
import os
import collections



def dominant_color_set(rgb_list, n_most = 1, rgb_buffers=(5, 5, 5)):
    """
    the functions return a list of dominant color (R,G,B) that charcterizes the map feature of interest
    < Arguments >
    * rgb_list: (r,b,g) list of n-most frequent colors (output of function "hex_to_rgb()")
    * n_most: the number of colors that would characterize the map feature of interest
    * rgb_buffers: R,G,B color buffer for color intervals considered featured color      
    """
    RGB_sets = [rgb for rgb, freq, prob in rgb_list[:n_most]]
    r_buffer, g_buffer, b_buffer = rgb_buffers 

    feature_colors = []
    for rgb in RGB_sets:
        R, G, B = rgb
        R_max, G_max, B_max = (R + r_buffer, G + g_buffer, B + b_buffer)
        R_min, G_min, B_min = (R - r_buffer, G - g_buffer, B - b_buffer)
        colors = ((R_min, G_min, B_min), (R_max, G_max, B_max))
        feature_colors.append(colors)        
    return feature_colors


def pic_val_count(img_name):
    """
    the function counts colors (R,G,B) of input image, and returns with frequency
    < Arguments >
    * img_nam: image file name, e.g.) 'image.png'
    """
    pic = cv2.imread(img_name)
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)

    reshaped_pic = np.reshape(pic, (pic.shape[0]*pic.shape[1], 3))
    reshaped_pic = reshaped_pic.tolist()
    reshaped_pic = [tuple(pixel) for pixel in reshaped_pic]
    
    col_count = []
    for i in set(reshaped_pic):
        (col_val, num_pic)  = i,  reshaped_pic.count(i)
        col_count.append((col_val, num_pic))        
    return col_count



def classify_feature_image(input_img, feature_colors, pix_cutoff=50):
    """
    the function detects color of interest from input image
    < Arguments >
    * input_img: image file name, e.g.) 'image.png'
    * feature_colors: a list of featured color obtained from "dominant_color_set()"
    * pix_cutoff: the threshold number of featured pixel to be considered 'positive' image
    """
    result = 'negative'
    for pic_val, num in pic_val_count(input_img):
        for min_rgb, max_rgb in feature_colors:
            if (((min_rgb[0] <= pic_val[0] <= max_rgb[0])
            &(min_rgb[1] <= pic_val[1] <= max_rgb[1])
            &(min_rgb[2] <= pic_val[2] <= max_rgb[2])) & (num > pix_cutoff)):
                result = "positive"
    return result