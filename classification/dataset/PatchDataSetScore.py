# -*- coding: utf-8 -*-
import os
import numpy as np
from tools.Tools import shuffle_image_label
from Config import Config
from tools.image_operations import tiffs_read

# 同样是基于Ｐａｔｃｈ做的，不过这里的ｌａｂｅｌ是得分，而不是第几类
class PatchDataSetBaseScore:
    def __init__(self, path, threshold=0.3, upthreshold=0.8, downthreshold=0.3):
        self.upthreshold=upthreshold
        self.downthreshold=downthreshold
        self.threshold = threshold
        self.path = path
        # 在这里之所以还要划分ｌａｂｅｌ是因为我们在构建ｐａｔｃｈ的时候为了平衡每个阶段的样本数目，所以需要ｌａｂｅｌ
        self.image_paths, self.scores, self.labels = self.load_patches_paths_labels(
            self.path
        )
        self.image_paths = np.array(self.image_paths)
        self.labels = np.array(self.labels)

    def get_next_batch(self, batch_size, distribution):
        patches = []
        scores = []
        for index, num in enumerate(distribution):
            indexs = np.where(self.labels == index)[0]
            np.random.shuffle(indexs)
            selected_index = indexs[:num]
            patches.extend(self.image_paths[selected_index])
            scores.extend(self.scores[selected_index])
        patches = np.array(patches)
        scores = np.array(scores)
        patches, scores = shuffle_image_label(patches, scores)
        patches_image = tiffs_read(patches)
        return patches_image, scores

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
        patch_scores = []
        for patch_name in patch_names:
            if patch_name.endswith('.txt'):
                continue
            patch_path = os.path.join(path, patch_name)
            patch_paths.append(patch_path)
            last_position = patch_name.find('.tiff')
            start_position = patch_name.rfind('_')
            occupy_rate = float(patch_name[start_position+1:last_position])
            patch_scores.append(occupy_rate)
        print '-'*15, 'load_patches_paths_labels finished.', '-'*15
        patch_scores = np.array(patch_scores)
        patch_labels = [self.get_label(x) for x in patch_scores]
        return patch_paths, patch_scores, patch_labels


class PatchDataSetScore:
    def __init__(self, config, threshold=0.3, up_threshold=0.8, down_threshold=0.4):
        self.config = config
        self.train_dataset = PatchDataSetBaseScore(self.config.PATCH_TRAIN_DATA_DIR, threshold, up_threshold, down_threshold)
        self.val_dataset = PatchDataSetBaseScore(self.config.PATCH_VAL_DATA_DIR, threshold, up_threshold, down_threshold)

    def get_next_train_batch(self, batch_size, batch_distribution):
        return self.train_dataset.get_next_batch(batch_size, batch_distribution)

    def get_next_val_batch(self, batch_size, batch_distribution):
        return self.val_dataset.get_next_batch(batch_size, batch_distribution)

if __name__ == '__main__':
    dataset = PatchDataSetScore(Config)
    images, labels = dataset.get_next_train_batch(40, [20, 20])
    print np.shape(images), np.shape(labels)
    print labels