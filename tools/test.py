# -*- coding=utf-8 -*-
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from classification.Net.ResNet.image_processing import image_preprocessing

def read_images(image_dir):
    names = os.listdir(image_dir)
    pathes = [os.path.join(image_dir, name) for name in names]
    for path in pathes:
        image = Image.open(path)
        print path, ' shape: ', np.shape(image)
def tensorflow_read(image_dir):
    names = os.listdir(image_dir)
    pathes = [os.path.join(image_dir, name) for name in names]
    path_tensor = tf.placeholder(
        tf.string
    )
    image_buffer = tf.read_file(path_tensor)
    bbox = []
    image = image_preprocessing(
        image_buffer,
        bbox=bbox,
        train=True,
        thread_id=0
    )
    print tf.shape(image)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for path in pathes:
            print path
            image_value = sess.run(image, {
                path_tensor:path
            })
            print np.shape(image_value)

'''
    人为的判断图像
'''
def mark_label(path, contained=None, record_path='/home/give/毕业的西瓜籽-V4.txt'):
    def show_image(path):
        img = Image.open(path)
        img.show()
        return img
    if contained is None:
        file_names = os.listdir(path)
        file_pathes = [os.path.join(path, file_name) for file_name in file_names]
    else:
        file_pathes = [os.path.join(path, file_name) for file_name in contained]
    record_dict = {}
    lines = None
    if record_path is not None:
        record_filed = open(record_path)
        lines = record_filed.readlines()
    lines = [line.replace('\n', '') for line in lines]
    index = 0
    for file_path in file_pathes:
        # print index, lines[index], file_path
        if lines is not None:
            if index < len(lines):
                value = int(lines[index][-1])
                print os.path.basename(file_path) + ':' + str(value)
            else:
                show_image(file_path)
                value = input(os.path.basename(file_path) + ':')
        else:
            show_image(file_path)
            value = input(os.path.basename(file_path) + ':')
        if value == 2:
            break
        if value == 1:
            record_dict[file_path] = 'P'
        else:
            record_dict[file_path] = 'N'
        index += 1
    filed = open('./毕业的西瓜籽-V4.txt', 'w')
    lines = []
    for key in record_dict.keys():
        line = os.path.basename(key) + ' ' + record_dict[key] + '\n'
        lines.append(line)
    filed.writelines(lines)

'''
    将ｐａｔｈ替换成ｂａｓｅｎａｍｅ
'''
def replace_path_basename(record_txt):
    filed = open(record_txt)
    lines = filed.readlines()
    filed.close()
    lines = [line.replace('\n', '') for line in lines]
    basenames = [os.path.basename(line.split(' ')[0]) for line in lines]
    new_lines = [basenames[index] + ' ' + line[-1] + '\n' for index, line in enumerate(lines)]
    new_lines = [line.replace('.jpg', '.tiff') for line in new_lines]
    filed = open(record_txt, 'w')
    filed.writelines(new_lines)
    filed.close()

'''
    统计预测的Ｐ和Ｎ各有多少个
'''
def static_pn(record_txt):
    filed = open(record_txt)
    lines = filed.readlines()
    filed.close()
    lines = [line.replace('\n', '') for line in lines]
    p_n = 0
    n_n = 0
    for line in lines:
        if line[-1] == 'P':
            p_n += 1
        elif line[-1] == 'N':
            n_n += 1
        else:
            print 'Error ', line
    print 'the number of positive %d, the number of negative %d' % (p_n, n_n)

'''
    依据我们手动标记的为ｇｒｏｕｎｄ　ｔｒｕｅ，计算f1ｓｃｏｒｅ
'''
def cal_f1_test(record_txt, predict_txt):
    def generate_dict(record_txt):
        filed = open(record_txt)
        lines = filed.readlines()
        filed.close()
        lines = [line.replace('\n', '') for line in lines]
        res_dict = {}
        for line in lines:
            basename = line.split(' ')[0]
            value = line.split(' ')[1]
            if value == 'P':
                value = 1
            elif value == 'N':
                value = 0
            else:
                print 'Error'
            res_dict[basename] = value
        return res_dict
    def dict2list(mydict):
        keys = list(mydict.keys())
        keys.sort()
        res = []
        for key in keys:
            res.append(mydict[key])
        return res
    gt = generate_dict(record_txt)
    predict = generate_dict(predict_txt)
    gt_list = dict2list(gt)
    predict_list = dict2list(predict)
    from Tools import get_game_evaluate
    recall, precision, f1_score = get_game_evaluate(predict_list, gt_list)
    print 'recall is %g, precision is %g, f1_score is %g\n' % (recall, precision, f1_score)
'''
    找出分类错误的
'''
def show_error_name(record_txt, predict_txt):
    def generate_dict(record_txt):
        filed = open(record_txt)
        lines = filed.readlines()
        filed.close()
        lines = [line.replace('\n', '') for line in lines]
        res_dict = {}
        for line in lines:
            basename = line.split(' ')[0]
            value = line.split(' ')[1]
            if value == 'P':
                value = 1
            elif value == 'N':
                value = 0
            else:
                print 'Error'
            res_dict[basename] = value
        return res_dict
    def dict2list(mydict):
        keys = list(mydict.keys())
        keys.sort()
        res = []
        for key in keys:
            res.append(mydict[key])
        return res
    gt = generate_dict(record_txt)
    predict = generate_dict(predict_txt)
    for key in gt.keys():
        if gt[key] != predict[key]:
            print key, ' not equal', ' gt is ', gt[key], ' predict is ', predict[key]

if __name__ == '__main__':
    # tensorflow_read('/home/give/Documents/dataset/BOT_Game/patches/method4/train/positive')
    # static_pn('/home/give/毕业的西瓜籽-V4.txt')
    show_error_name('/home/give/毕业的西瓜籽-V4.txt', '/home/give/PycharmProjects/StomachCanner/classification/Net/ResNet/毕业的西瓜籽-V3-0.8.txt')