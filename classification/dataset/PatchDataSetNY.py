import os
import numpy as np
from tools.Tools import shuffle_image_label
from Config import Config
from tools.image_operations import tiffs_read


class PatchDataSetNYBase:
    def __init__(self, data_path):
        self.data_path = data_path
        self.positive_data, self.positive_label = PatchDataSetNYBase.load_patches_paths_labels(
            os.path.join(data_path, 'positive'),
            1
        )
        self.negative_data, self.negative_label = PatchDataSetNYBase.load_patches_paths_labels(
            os.path.join(data_path, 'negative'),
            1
        )
        self.data, self.label = PatchDataSetNYBase.merged_PN(
            self.positive_data,
            self.positive_label,
            self.negative_data,
            self.negative_label
        )
        self.start_index = 0
        self.epoch_num = 0

    @staticmethod
    def load_patches_paths_labels(path, label):
        patch_names = os.listdir(path)
        patch_paths = []
        patch_labels = []
        for patch_name in patch_names:
            if patch_name.endswith('.txt'):
                continue
            patch_path = os.path.join(path, patch_name)
            patch_paths.append(patch_path)
            patch_labels.append(label)
        patches = []
        patches.extend(
            tiffs_read(patch_paths, [224, 224])
        )
        return patches, patch_labels

    @staticmethod
    def merged_PN(pdata, plabel, ndata, nlabel):
        data = []
        label = []
        data.extend(pdata)
        label.extend(plabel)
        data.extend(ndata)
        label.extend(nlabel)
        index = range(len(data))
        np.random.shuffle(index)
        data = np.array(data)
        label = np.array(label)
        data = data[index]
        label = label[index]
        return data, label

    def get_next_batch(self, batch_size):
        end_index = self.start_index + batch_size
        if end_index > len(self.data):
            end_index = len(self.data)
        batch_data = self.data[self.start_index:end_index]
        # print end_index
        labels = self.label[self.start_index:end_index]
        if end_index == len(self.data):
            self.start_index = 0
            self.epoch_num += 1
            print '-'*15, 'epoch num is ', self.epoch_num, 'data len is ', len(self.data), '-'*15
        else:
            self.start_index = end_index
        return batch_data, labels

    def get_average_image(self):
        images = None
        for image in self.data:
            if images is None:
                images = image
            else:
                images += image
        return images/len(self.data)

class PatchDataSetNY:
    def __init__(self, config=Config):
        self.train_data = PatchDataSetNYBase(
            config.PATCH_TRAIN_DATA_DIR_NY
        )
        self.val_data = PatchDataSetNYBase(
            config.PATCH_VAL_DATA_DIR_NY
        )

    def get_next_train_batch(self, batch_size):
        return self.train_data.get_next_batch(batch_size)

    def get_next_val_batch(self, batch_size):
        return self.val_data.get_next_batch(batch_size)
