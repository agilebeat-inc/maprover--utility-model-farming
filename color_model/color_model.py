from itertools import compress 
import numpy as np
import cv2
import base64
import io

#check hear first!
def pic_val_count(pic_RGB):
    """
    """
    reshaped_pic = np.reshape(pic_RGB, (pic_RGB.shape[0]*pic_RGB.shape[1], 3))
    reshaped_pic = reshaped_pic.tolist()
    reshaped_pic = [tuple(pixel) for pixel in reshaped_pic]
    
    col_count = []
    for i in set(reshaped_pic):
        (col_val, num_pic)  = i,  reshaped_pic.count(i)
        col_count.append((col_val, num_pic))        
    return col_count

def read_pixel_value(output_file):
    with open(output_file, 'r+') as f:
        col_vals = f.read().splitlines()
    col_vals_list = [int(val) for val in col_vals]
    return col_vals

def decode_base64_to_cv2(img_b64):
    img = base64.urlsafe_b64decode(img_b64)
    img_io = io.BytesIO(img)
    img_np = np.frombuffer(img_io.read(), dtype=np.uint8)
    img_cv2 = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    pic_RGB = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    return pic_RGB  # 256x256x3 
    
def classify_feature_image_b64(input_img_b64, rgb_list, pix_cutoff=50):
    """
    """
    pic_RGB = decode_base64_to_cv2(input_img_b64)
    return classify_feature_image(pic_RGB, rgb_list, pix_cutoff)


def classify_feature_image(pic_RGB, rgb_list, pix_cutoff=50):
    """
    """
    result = 'negative'
    min_R, max_R, min_G, max_G, min_B, max_B = compress(rgb_list, (1, 1, 1, 1, 1, 1)) 
    
    for pic_val, num in pic_val_count(pic_RGB):
        if ((min_R <= pic_val[0] <= max_R)
            &(min_G <= pic_val[1] <= max_G)
            &(min_B <= pic_val[2] <= max_B)
            &(num > pix_cutoff)):
                result = "positive"
    return result