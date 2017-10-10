# -*- coding=utf-8 -*-
import os
from PIL import Image
import numpy as np

def generate_restxt_heatmap(res_dir, res_txt_path):
    def cal_sum(path):
        image = np.array(Image.open(path))
        return np.sum(image)
    names = os.listdir(res_dir)
    name_pathes = [os.path.join(res_dir, name) for name in names]
    sums = [cal_sum(path) for path in name_pathes]
    opened_file = open(res_txt_path, 'w')
    lines = []
    for index, path in enumerate(name_pathes):
        basename = os.path.basename(path).split('.png')[0] + '.tiff'
        if sums[index] == 0:
            flag = 'N'
        else:
            flag = 'P'
        lines.append(basename+' ' + flag + '\n')
    opened_file.writelines(lines)
    opened_file.close()
    print sums

if __name__ == '__main__':
    generate_restxt_heatmap('/home/give/Documents/dataset/BOT_Game/0-testdataset-hm',
                            '/home/give/PycharmProjects/StomachCanner/classification/Net/ResNet/毕业的西瓜籽-V2.txt')