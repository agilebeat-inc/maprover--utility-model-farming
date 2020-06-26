# pylint: disable=line-too-long
""" Module provides an image abstraction
"""
import io
import base64
import numpy as np
import cv2


class Image:
    """Abstraction for an image independet from storgae
    """
    def __init__(self, pic, rgbs, hexs):
        self.pic = pic
        self.rgbs = rgbs
        self.hexs = hexs

    def save(self, filename):
        """Persists image to a file system

        Args:
            filename (path): Full path where image will ba saved
        """
        cv2.imwrite(filename, self.pic)

    @staticmethod
    def compute_representations(pic):
        """Factory method

        Args:
            pic ([type]): [description]

        Returns:
            [type]: [description]
        """

        pic_rgb = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        reshaped_pic = np.reshape(
            pic_rgb, (pic_rgb.shape[0] * pic_rgb.shape[1], 3)).tolist()
        rgbs = [(pixel[0], pixel[1], pixel[2]) for pixel in reshaped_pic]
        hexs = ['%02x%02x%02x' % rgb for rgb in rgbs]
        return rgbs, hexs

    @staticmethod
    def load_from_filesystem(path):
        """[summary]

        Args:
            path ([type]): [description]

        Returns:
            [type]: [description]
        """
        pic = cv2.imread(path)
        rgbs, hexs = Image.compute_representations(pic)
        return Image(pic, rgbs, hexs)

    @staticmethod
    def load_from_b64string(img_b64):
        """[summary]

        Args:
            img_b64 ([type]): [description]

        Returns:
            [type]: [description]
        """
        img = base64.urlsafe_b64decode(img_b64)
        img_io = io.BytesIO(img)
        img_np = np.frombuffer(img_io.read(), dtype=np.uint8)
        pic = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        rgbs, hexs = Image.compute_representations(pic)
        return Image(pic, rgbs, hexs)
