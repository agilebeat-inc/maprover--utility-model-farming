# pylint: disable=line-too-long
""" Basic clolor model allows to check tile content if the given color is present.
"""

import base64

import collections
import io
import json
import time
import cv2
import numpy as np


class BasicColorModel:
    """Allows to classify tiles based on color only
    """

    def __init__(self, rgb_list, pix_cutoff=50):
        if not (isinstance(rgb_list, list)
                or isinstance(rgb_list, tuple)) or len(rgb_list) != 6:
            raise ValueError(
                'The model requires exactly six arguments in list: [min_r, max_r, min_g, max_g, min_v, max_v]'
            )
        self.timestamp = time.time()
        self.min_r = rgb_list[0]
        self.max_r = rgb_list[1]
        self.min_g = rgb_list[2]
        self.max_g = rgb_list[3]
        self.min_b = rgb_list[4]
        self.max_b = rgb_list[5]
        self.pix_cutoff = pix_cutoff

    def pic_val_count(self, pic_rgb):
        """[summary]

        Args:
            pic_rgb ([type]): [description]

        Returns:
            [type]: [description]
        """
        reshaped_pic = np.reshape(pic_rgb,
                                  (pic_rgb.shape[0] * pic_rgb.shape[1], 3))
        reshaped_pic = reshaped_pic.tolist()
        reshaped_pic = [tuple(pixel) for pixel in reshaped_pic]

        col_count = []
        for i in set(reshaped_pic):
            (col_val, num_pic) = i, reshaped_pic.count(i)
            col_count.append((col_val, num_pic))
        return col_count

    def decode_base64_to_cv2(self, img_b64):
        """[summary]

        Args:
            img_b64 ([type]): [description]

        Returns:
            [type]: [description]
        """
        img = base64.urlsafe_b64decode(img_b64)
        img_io = io.BytesIO(img)
        img_np = np.frombuffer(img_io.read(), dtype=np.uint8)
        img_cv2 = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        pic_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        return pic_rgb  # 256x256x3

    def predict_b64(self, input_img_b64, pix_cutoff=50):
        """[summary]

        Args:
            input_img_b64 ([type]): [description]
            pix_cutoff (int, optional): [description]. Defaults to 50.

        Returns:
            [type]: [description]
        """
        pic_rgb = self.decode_base64_to_cv2(input_img_b64)
        return self.predict(pic_rgb, pix_cutoff)

    def predict(self, pic_rgb, pix_cutoff=None):
        """[summary]

        Args:
            pic_RGB ([type]): [description]
            pix_cutoff ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        result = 'negative'

        if pix_cutoff is not None:
            pcf = pix_cutoff
        else:
            pcf = self.pix_cutoff

        for pic_val, num in self.pic_val_count(pic_rgb):
            if ((self.min_r <= pic_val[0] <= self.max_r)
                    & (self.min_g <= pic_val[1] <= self.max_g)
                    & (self.min_b <= pic_val[2] <= self.max_b)
                    & (num > pcf)):
                result = "positive"
        return result

    def to_json(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return json.dumps(self, default=lambda o: o.__dict__)

    @staticmethod
    def load(path):
        """[summary]

        Args:
            path ([type]): [description]

        Returns:
            [type]: [description]
        """
        with open(path) as json_file:
            model_json = json.load(json_file)
            min_r = model_json['min_r']
            max_r = model_json['max_r']
            min_g = model_json['min_g']
            max_g = model_json['max_g']
            min_b = model_json['min_b']
            max_b = model_json['max_b']
            pix_cutoff = model_json['pix_cutoff']
            return BasicColorModel([min_r, max_r, min_g, max_g, min_b, max_b],
                                   pix_cutoff=pix_cutoff)
        return None

    def save(self, path):
        """[summary]

        Args:
            path ([type]): [description]
        """
        with open(path, 'w+') as jeson_f:
            jeson_f.write(self.to_json())

    def __eq__(self, other):
        """[summary]

        Args:
            other ([type]): [description]

        Returns:
            [type]: [description]
        """
        if not other:
            return False
        if (self.max_b == other.max_b and self.max_g == other.max_g
                and self.max_r == other.max_r and self.min_b == other.min_b
                and self.min_g == other.min_g and self.min_r == other.min_r
                and self.pix_cutoff == other.pix_cutoff):
            return True
        return False

    @staticmethod
    def compute_color_dist(iterator, descending=True):
        """[summary]

        Args:
            iterator ([type]): [description]
            descending (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        color_vals = []
        valid_img = 0
        for _, image in iterator:
            hexs = image.hexs
            color_vals = color_vals + list(set(hexs))
            valid_img += 1

        freq = collections.Counter(color_vals)
        freq_sorted = sorted(freq.items(),
                             reverse=descending,
                             key=lambda item: item[1])
        freq_dist = [(hex_code, freq, round(freq / valid_img, 3))
                     for hex_code, freq in freq_sorted]
        return freq_dist

    @staticmethod
    def hex_to_rgb(hexs_freq, n_most_rgb=10):
        """[summary]

        Args:
            HEXs_Freq ([type]): [description]
            n_most_rgb (int, optional): [description]. Defaults to 10.

        Returns:
            [type]: [description]
        """
        rgb_list = []
        for hex_code, freq, pct in hexs_freq[:n_most_rgb]:
            value = hex_code.lstrip('#')
            lngth = len(value)
            rgb = tuple(
                int(value[i:i + lngth // 3], 16) for i in range(0, lngth, lngth // 3))
            rgb_list = rgb_list + [(rgb, freq, pct)]
        return rgb_list

    @staticmethod
    def dominant_color_set(rgb_list, n_most=1, rgb_buffers=(5, 5, 5)):
        """[summary]

        Args:
            rgb_list ([type]): [description]
            n_most (int, optional): [description]. Defaults to 1.
            rgb_buffers (tuple, optional): [description]. Defaults to (5, 5, 5).

        Returns:
            [type]: [description]
        """
        rgb_sets = [rgb for rgb, freq, prob in rgb_list[:n_most]]
        r_buffer, g_buffer, b_buffer = rgb_buffers

        feature_colors = []
        for rgb in rgb_sets:
            red, green, blue = rgb
            r_max, g_max, b_max = (red + r_buffer, green + g_buffer, blue + b_buffer)
            r_min, g_min, b_min = (red - r_buffer, green - g_buffer, blue - b_buffer)
            colors = (r_min, g_min, b_min, r_max, g_max, b_max)
            feature_colors.append(colors)
        return feature_colors

    @classmethod
    def color_set_generator(cls, iterator, rgb_buffers=(5, 5, 5)):
        """[summary]

        Args:
            iterator ([type]): [description]
            rgb_buffers (tuple, optional): [description]. Defaults to (5, 5, 5).

        Returns:
            [type]: [description]
        """
        hex_dist = cls.compute_color_dist(iterator, descending=True)
        rgb_list = cls.hex_to_rgb(hex_dist, n_most_rgb=1)
        feature_colors = cls.dominant_color_set(rgb_list,
                                                n_most=1,
                                                rgb_buffers=rgb_buffers)
        return feature_colors

    @classmethod
    def fit(cls, iterator):
        """[summary]

        Args:
            iterator ([type]): [description]

        Returns:
            [type]: [description]
        """
        feature_colors = cls.color_set_generator(iterator)
        return BasicColorModel(feature_colors[0])
