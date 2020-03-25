#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:46:23 2020

@author: swilson
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os





def TemplateMatch(templates, img, threshold):
    """
    the function returns if imput images contains template image 
    < Arguments >
    * templates: the path of the directory where template images are
    * img: input image. image file name, (e.g. 'image.png')
    * threshold: matching accuracy (between 0 and 1) 
    """
    prediction = 'positive'
    templates = [os.path.join(templates, template) for template in os.listdir(templates)]
    
    img_rgb = cv.imread(img)
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)  
    
    n_th_template = 0
    for tmpl in templates:
        template = cv.imread(tmpl, 0)
        w, h = template.shape[::-1]
        res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
        thresh = threshold
        
        loc = np.where(res >= thresh)
        if (len(loc[0]) == 0):
            pass
        else:
            break
        
    if (n_th_template == len(templates)):
        prediction = 'negative'
    return prediction
        

