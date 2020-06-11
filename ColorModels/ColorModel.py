import numpy as np
import cv2
import base64
import io
import json
import time


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

    def classify_feature_image_b64(self, input_img_b64, pix_cutoff=50):
        """
        """
        pic_RGB = self.decode_base64_to_cv2(input_img_b64)
        return self.classify_feature_image(pic_RGB, pix_cutoff)

    def classify_feature_image(self, pic_RGB, pix_cutoff=None):
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
