import os
from Image import Image


class FSIterator:
    def __init__(self, path, filter):
        self.records = self.__create_record_list(path, filter)
        self.index = 0

    def __iter__(self, ):
        return self

    def __next__(self):
        if self.index >= len(self.records):
            raise StopIteration
        ret_item = self.records[self.index]
        self.index += 1
        return ret_item

    def __create_record_list(self, path, filter):
        listOfFiles = list()
        for (relpath, dirnames, filenames) in os.walk(path):
            listOfFiles += [
                (os.path.basename(relpath),
                 Image.load_from_filesystem(os.path.join(relpath, file)))
                for file in filenames if filter == os.path.splitext(file)[1]
            ]
        return listOfFiles


class B64Iterator:
    def __init__(self, b64_imglst, class_name=None):
        self.class_name = class_name
        self.index = 0
        self.records = self.__create_record_list(b64_imglst)

    def __iter__(self, ):
        return self

    def __next__(self):
        if self.index >= len(self.records):
            raise StopIteration
        ret_item = self.records[self.index]
        self.index += 1
        return ret_item

    def __create_record_list(self, b64_imglst):
        listOfImages = list()
        for i_b64 in b64_imglst:
            listOfImages.append(
                (self.class_name, Image.load_from_b64string(i_b64)))
        return listOfImages