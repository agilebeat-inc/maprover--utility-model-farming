import feature_col_gen as col

import json
import os
import base64
import io
import cv2
import numpy as np

###== (STEP 1) Create feature color values and save output as 'file.mod'

# Sepecify directory of positive tiles & output file include path
dir_pos = '/workspaces/maprover--utility-model-farming/example-Color-based/dataset-color-based/landuse_construction/color_dist'
output_file ='test_construction.mod'

featured_color_value = col.color_set_generator(dir_pos, output_file, rgb_buffers=(3,3,3)) 


###== (STEP 2) Create feature color values and save as 'file.mod'

#--- call the feature color list created from traning 
output_file = 'test_construction.mod'
f = open(output_file, "r")
col_vals = f.read().splitlines()
f.close()

col_vals = [int(val) for val in col_vals]
#--- pixel values of the map feature
min_R, max_R = col_vals[0], col_vals[1] 
min_G, max_G = col_vals[2], col_vals[3]  
min_B, max_B = col_vals[4], col_vals[5] 


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


def classify_feature_image(input_img, pix_cutoff=50):
    """
    the function detects color of interest from input image
    < Arguments >
    * input_img: image file name, e.g.) 'image.png'
    * feature_colors: a list of featured color obtained from "dominant_color_set()"
    * pix_cutoff: the threshold number of featured pixel to be considered 'positive' image
    """
    result = 'negative'
    for pic_val, num in pic_val_count(input_img):
        if ((min_R <= pic_val[0] <= max_R)
            &(min_G <= pic_val[1] <= max_G)
            &(min_B <= pic_val[2] <= max_B)
            &(num > pix_cutoff)):
                result = "positive"
    return result



#--- TEST
test_img = '/workspaces/maprover--utility-model-farming/example-Color-based/dataset-color-based/landuse_construction/TEST/construction/19_432079_197852.png'
classify_feature_image(test_img, pix_cutoff = 50)
