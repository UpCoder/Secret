import os
import numpy as np
from tools.Tools import shuffle_image_label
from Config import Config
from tools.image_operations import tiffs_read


class PatchDataSetBase:
    def __init__(self, path, threshold=0.3, upthreshold=0.8, downthreshold=0.3, using_border=True):
        self.upthreshold=upthreshold
        self.downthreshold=downthreshold
        self.threshold = threshold
        self.path = path
        self.using_border = using_border
        self.image_paths, self.labels = self.load_patches_paths_labels(
            self.path
        )
        self.image_paths = np.array(self.image_paths)
        self.labels = np.array(self.labels)

    def get_next_batch(self, batch_size, distribution):
        patches = []
        labels = []
        for index, num in enumerate(distribution):
            indexs = np.where(self.labels == index)[0]
            np.random.shuffle(indexs)
            selected_index = indexs[:num]
            labels.extend(self.labels[selected_index])
            patches.extend(self.image_paths[selected_index])
        patches = np.array(patches)
        labels = np.array(labels)
        patches, labels = shuffle_image_label(patches, labels)
        patches_image = tiffs_read(patches)
        return patches_image, labels

    # get label
    def get_label(self, occupy_rate):
        if Config.OUTPUT_NODE == 3:
            if occupy_rate > self.upthreshold:
                return 2
            if occupy_rate > self.downthreshold:
                return 1
            return 0
        elif Config.OUTPUT_NODE == 2:
            if occupy_rate > self.threshold:
                return 1
            return 0

    def load_patches_paths_labels(self, path):
        patch_names = os.listdir(path)
        patch_paths = []
        patch_labels = []
        for patch_name in patch_names:
            if patch_name.endswith('.txt'):
                continue
            patch_path = os.path.join(path, patch_name)
            last_position = patch_name.find('.tiff')
            start_position = patch_name.rfind('_')
            occupy_rate = float(patch_name[start_position+1:last_position])
            if not self.using_border:
                if occupy_rate < self.threshold and \
                        not (patch_name.startswith('Normal') or patch_name.startswith('normal')):
                    continue
            patch_paths.append(patch_path)
            patch_labels.append(self.get_label(occupy_rate))
        print '-'*15, 'load_patches_paths_labels finished.', '-'*15
        patch_labels = np.array(patch_labels)
        print '0 is ', np.sum(patch_labels == 0), \
            ' 1 is ', np.sum(patch_labels == 1), \
            '2 is ', np.sum(patch_labels == 2)
        return patch_paths, patch_labels


class PatchDataSet:
    def __init__(self, config, threshold=0.3, up_threshold=0.8, down_threshold=0.4, using_border=True):
        self.config = config
        self.train_dataset = PatchDataSetBase(self.config.PATCH_TRAIN_DATA_DIR, threshold, up_threshold, down_threshold,
                                              using_border=using_border)
        self.val_dataset = PatchDataSetBase(self.config.PATCH_VAL_DATA_DIR, threshold, up_threshold, down_threshold,
                                            using_border=using_border)

    def get_next_train_batch(self, batch_size, batch_distribution):
        return self.train_dataset.get_next_batch(batch_size, batch_distribution)

    def get_next_val_batch(self, batch_size, batch_distribution):
        return self.val_dataset.get_next_batch(batch_size, batch_distribution)

if __name__ == '__main__':
    dataset = PatchDataSet(Config)
    images, labels = dataset.get_next_train_batch(40, [20, 20])
    print np.shape(images), np.shape(labels)