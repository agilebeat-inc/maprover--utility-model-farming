import numpy as np
import cv2
import base64
import io
import json
import time
import os
import collections
from .Iterator import FSIterator


class BasicColorModel:
    def __init__(self, rgb_list, pix_cutoff=50):
        if not isinstance(rgb_list, list) or len(rgb_list) != 6:
            raise ValueError(
                'The model requires exactly six arguments in list: [min_R, max_R, min_G, max_G, min_B, max_B]')
        self.timestamp = time.time()
        self.min_R = rgb_list[0]
        self.max_R = rgb_list[1]
        self.min_G = rgb_list[2]
        self.max_G = rgb_list[3]
        self.min_B = rgb_list[4]
        self.max_B = rgb_list[5]
        self.pix_cutoff = pix_cutoff

    def pic_val_count(self, pic_RGB):
        """
        """
        reshaped_pic = np.reshape(
            pic_RGB, (pic_RGB.shape[0]*pic_RGB.shape[1], 3))
        reshaped_pic = reshaped_pic.tolist()
        reshaped_pic = [tuple(pixel) for pixel in reshaped_pic]

        col_count = []
        for i in set(reshaped_pic):
            (col_val, num_pic) = i,  reshaped_pic.count(i)
            col_count.append((col_val, num_pic))
        return col_count

    def read_pixel_value(self, output_file):
        with open(output_file, 'r+') as f:
            col_vals = f.read().splitlines()
        col_vals_list = [int(val) for val in col_vals]
        return col_vals

    def decode_base64_to_cv2(self, img_b64):
        img = base64.urlsafe_b64decode(img_b64)
        img_io = io.BytesIO(img)
        img_np = np.frombuffer(img_io.read(), dtype=np.uint8)
        img_cv2 = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        pic_RGB = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        return pic_RGB  # 256x256x3

    def predict_b64(self, input_img_b64, pix_cutoff=50):
        """
        """
        pic_RGB = self.decode_base64_to_cv2(input_img_b64)
        return self.predict(pic_RGB, pix_cutoff)

    def predict(self, pic_RGB, pix_cutoff=None):
        """
        """
        result = 'negative'

        if pix_cutoff is not None:
            pc = pix_cutoff
        else:
            pc = self.pix_cutoff

        for pic_val, num in self.pic_val_count(pic_RGB):
            if ((self.min_R <= pic_val[0] <= self.max_R)
                & (self.min_G <= pic_val[1] <= self.max_G)
                & (self.min_B <= pic_val[2] <= self.max_B)
                    & (num > pc)):
                result = "positive"
        return result

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__)

    @staticmethod
    def load(path):
        with open(path) as json_file:
            model_json = json.load(json_file)
            timestamp = model_json['timestamp']
            min_R = model_json['min_R']
            max_R = model_json['max_R']
            min_G = model_json['min_G']
            max_G = model_json['max_G']
            min_B = model_json['min_B']
            max_B = model_json['max_B']
            pix_cutoff = model_json['pix_cutoff']
            return BasicColorModel([min_R, max_R,
                                    min_G, max_G,
                                    min_B, max_B],
                                   pix_cutoff=pix_cutoff)
        return None

    def save(self, path):
        with open(path, 'w+') as jf:
            jf.write(self.toJson())

    def __eq__(self, other):
        if not other:
            return False
        if (self.max_B == other.max_B and
            self.max_G == other.max_G and
            self.max_R == other.max_R and
            self.min_B == other.min_B and
            self.min_G == other.min_G and
            self.min_R == other.min_R and
            self.pix_cutoff == other.pix_cutoff):
           return True
        return False

@staticmethod
def compute_color_dist(pos_dir, descending=True):
    f_iterator = FSIterator(pos_dir, '.png')
    
    color_vals = []  
    valid_img = 0 
    for rel_root, f, f_ext, f_size in f_iterator:
        if f_size <= 0:
            continue
        try:
            pic = cv2.imread(os.path.join(rel_root, f))
            pic_RGBArr = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
                
            reshaped_pic = np.reshape(pic_RGBArr, (pic_RGBArr.shape[0]*pic_RGBArr.shape[1], 3))
            reshaped_pic = reshaped_pic.tolist()
                
            RGBs = [(pixel[0], pixel[1], pixel[2]) for pixel in reshaped_pic]
            HEXs = ['%02x%02x%02x' % rgb for rgb in RGBs]
            color_vals = color_vals + list(set(HEXs))
            valid_img += 1
        except:
            continue
        
    total_n_images = valid_img
    Freq = collections.Counter(color_vals)
    Freq = {k: v for k, v in sorted(Freq.items(), 
                                    reverse=descending, key=lambda item: item[1])}
    HEXs_Freq = list(Freq.items())   
    HEXs_Freq = [(hex_code, freq, round(freq/total_n_images, 3) ) 
                     for hex_code, freq in HEXs_Freq]
    return HEXs_Freq 

    @staticmethod
    def hex_to_rgb(HEXs_Freq, n_most_rgb=10):
        rgb_list = []    
        for hex_code, freq, pct in HEXs_Freq[:n_most_rgb]:
            value = hex_code.lstrip('#')
            lv = len(value)
            rgb = tuple(int(value[i:i+lv//3], 16) for i in range(0, lv, lv//3))
            rgb_list = rgb_list + [(rgb, freq, pct)]
        return rgb_list
    
    @staticmethod
    def dominant_color_set(rgb_list, n_most = 1, rgb_buffers=(5, 5, 5)):
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

    @classmethod
    def color_set_generator(cls, dir_pos, output_file, rgb_buffers=(5,5,5)):
        HEXs_Freq = cls.compute_color_dist(dir_pos, descending=True)
        rgb_list = cls.hex_to_rgb(HEXs_Freq, n_most_rgb=10)
        feature_colors = cls.dominant_color_set(rgb_list, n_most = 1, 
                                                rgb_buffers=rgb_buffers)
        return feature_colors

    @classmethod
    def fit(cls, iterator):
        cls.color_set_generator()