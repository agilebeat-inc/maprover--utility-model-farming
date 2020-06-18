import cv2
import numpy as np
import base64
import io

class Image:
    def __init__(self, pic, RGBs, HEXs):
        self.pic = pic
        self.RGBs = RGBs
        self.HEXs = HEXs

    def save(self, filename):
        print(filename)
        cv2.imwrite(filename, self.pic)

    @staticmethod
    def compute_representations(pic):
        pic_RGB = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        reshaped_pic = np.reshape(pic_RGB, (pic_RGB.shape[0]*pic_RGB.shape[1], 3)).tolist()
        RGBs = [(pixel[0], pixel[1], pixel[2]) for pixel in reshaped_pic]
        HEXs = ['%02x%02x%02x' % rgb for rgb in RGBs]
        return RGBs, HEXs

    @staticmethod
    def load_from_filesystem(path):
        pic = cv2.imread(path)
        RGBs, HEXs = Image.compute_representations(pic)
        return Image(pic, RGBs, HEXs)

    @staticmethod
    def load_from_b64string(img_b64):
        img = base64.urlsafe_b64decode(img_b64)
        img_io = io.BytesIO(img)
        img_np = np.frombuffer(img_io.read(), dtype=np.uint8)
        pic = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        RGBs, HEXs = Image.compute_representations(pic)
        return Image(pic, RGBs, HEXs)
