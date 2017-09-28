# -*- coding: utf-8 -*-
from Config import Config
from tools.image_operations import tiff_read, tiffs_read
import os
import numpy as np
from tools.Tools import shuffle_image_label


class DataSetBase:
    def __init__(self, path):
        self.path = path
        self.train_image_names = {}
        self.train_image_names['positive'] = DataSetBase.load_image_paths(
            os.path.join(path, 'positive')
        )
        self.train_image_names['negative'] = DataSetBase.load_image_paths(
            os.path.join(path, 'negative')
        )
        # self.val_image_names = {}
        # self.val_image_names['positive'] = DataSetBase.load_image_paths(
        #     os.path.join(path, 'positive')
        # )
        # self.val_image_names['negative'] = DataSetBase.load_image_paths(
        #     os.path.join(path, 'negative')
        # )
        self.images, self.labels = self.merge_train_val()
        self.start_index_merged = 0
        self.start_index = 0

    # 获取文件夹下面的所有文件名称
    @staticmethod
    def load_image_paths(path):
        file_names = os.listdir(path)
        paths = []
        for file_name in file_names:
            paths.append(
                os.path.join(path, file_name)
            )
        return paths

    def mered_has_next(self):
        if self.start_index_merged < len(self.images):
            return True
        return False

    # 获取政府样本混合的下一个ｂａｔｃｈ
    def merged_get_next(self, batch_size):
        end_index = self.start_index_merged + batch_size
        flag = True
        if end_index >= len(self.images):
            end_index = len(self.images)
            flag = False
        image_names = self.images[self.start_index_merged:end_index]
        labels = self.labels[self.start_index_merged:end_index]
        self.start_index_merged = end_index
        images = tiffs_read(image_names)
        return images, labels, flag

    # 获取下一个ｂａｔｃｈ数据
    def get_next_batch(self, batch_size, batch_distribution=None):
        images = []
        labels = []
        if batch_distribution is None:
            print 'batch distribution is None'
            return images, labels
        for _ in range(batch_distribution[0]):
            # positive
            random_index = np.random.randint(len(self.train_image_names['positive']))
            images.append(
                tiff_read(
                    self.train_image_names['positive'][random_index]
                )
            )
            labels.append(1)
        for _ in range(batch_distribution[1]):
            # negative
            random_index = np.random.randint(len(self.train_image_names['negative']))
            images.append(
                tiff_read(
                    self.train_image_names['negative'][random_index]
                )
            )
            labels.append(0)
        images = np.array(images)
        labels = np.array(labels)
        images, labels = shuffle_image_label(images, labels)
        return images, labels

    def merge_train_val(self):
        images = []
        labels = []
        images.extend(self.train_image_names['positive'])
        labels.extend([1] * len(self.train_image_names['positive']))
        images.extend(self.train_image_names['negative'])
        labels.extend([0] * len(self.train_image_names['negative']))
        images, labels = shuffle_image_label(images, labels)
        return images, labels

class DataSet:
    def __init__(self, config):
        self.train_dataset = DataSetBase(config.TRAIN_DATA_DIR)
        self.val_dataset = DataSetBase(config.VAL_DATA_DIR)

    def get_next_train_batch(self, batch_size, batch_distribution):
        return self.train_dataset.get_next_batch(batch_size, batch_distribution)

    def get_next_val_batch(self, batch_size, batch_distribution):
        return self.val_dataset.get_next_batch(batch_size, batch_distribution)

    def get_val_has_next(self):
        return self.val_dataset.mered_has_next()

    def get_val_merged_next_batch(self, batch_size):
        return self.val_dataset.merged_get_next(batch_size)

    def get_val_merged_image_name(self):
        return self.val_dataset.images
if __name__ == '__main__':
    dataset = DataSetBase(Config.TRAIN_DATA_DIR)
    images, labels = dataset.get_next_batch(Config.TRAIN_BATCH_SIZE, Config.TRAIN_BATCH_DISTRIBUTION)
    print np.shape(images)
