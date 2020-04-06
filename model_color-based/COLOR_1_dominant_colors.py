#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:30:26 2020

@author: swilson
"""

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import collections


def color_dist(dir_pos, descending=True):
    """
    the function returns list of tuples (hex_code, freq) in descending (default) order of frequency  
    < Arguments > 
    * dir_pos: path of directory where postivie images are       
    """
    tiles = [os.path.join((dir_pos), file)  for file in os.listdir(dir_pos)]
    
    color_vals = []   
    for img in tiles:
        pic = cv2.imread(img)
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        
        reshaped_pic = np.reshape(pic, (pic.shape[0]*pic.shape[1], 3))
        reshaped_pic = reshaped_pic.tolist()
        
        RGBs = [(pixel[0], pixel[1], pixel[2]) for pixel in reshaped_pic]
        HEXs = ['%02x%02x%02x' % rgb for rgb in RGBs]
        color_vals = color_vals + list(set(HEXs))
        
    total_n_images = len(tiles)
    Freq = collections.Counter(color_vals)
    Freq = {k: v for k, v in sorted(Freq.items(), 
                                    reverse=descending, key=lambda item: item[1])}
    HEXs_Freq = list(Freq.items())   
    HEXs_Freq = [(hex_code, freq, round(freq/total_n_images, 3) ) 
                     for hex_code, freq in HEXs_Freq]
    return HEXs_Freq 


def hex_to_rgb(HEXs_Freq, n_most_rgb=10):
    """
    the function converts HEXs to RGB code for n-most frequent color used in positive data
    < Arguments >
    * HEXs_Freq: the list of HEXs color codes collected from the positive data
    * n_most_rgb: limites output. Returns n colors only (descending order)
    """
    rgb_list = []    
    for hex_code, freq, pct in HEXs_Freq[:n_most_rgb]:
        value = hex_code.lstrip('#')
        lv = len(value)
        rgb = tuple(int(value[i:i+lv//3], 16) for i in range(0, lv, lv//3))
        rgb_list = rgb_list + [(rgb, freq, pct)]
    return rgb_list






    