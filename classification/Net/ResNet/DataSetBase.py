import os
from tools.image_operations import load_image_names


class DataSetBase:
    def __init__(self, positive_path=None, negative_path=None):
        self.images_names = []
        self.labels = []
        if positive_path is not None:
            positive_names = load_image_names(positive_path)
            self.labels.extend([1] * len(positive_names))
            self.images_names.extend(positive_names)
        if negative_path is not None:
            negative_names = load_image_names(negative_path)
            self.labels.extend([0]*len(negative_names))
            self.images_names.extend(negative_names)