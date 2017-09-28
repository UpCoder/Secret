# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import shutil
import os


# 将数据打乱
def shuffle_image_label(images, labels):
    images = np.array(images)
    labels = np.array(labels)
    random_index = range(len(images))
    np.random.shuffle(random_index)
    images = images[random_index]
    labels = labels[random_index]
    return images, labels


# 计算Ａｃｃｕｒａｃｙ，并且返回每一类最大错了多少个
def calculate_acc_error(logits, label, show=True, images_name=None):
    error_dict = {}
    error_dict_record = {}
    error_count = 0
    error_names = []
    # print logits
    for index, logit in enumerate(logits):
        if logit != label[index]:
            if images_name is not None:
                error_names.append(
                    images_name[index]
                )
            error_count += 1
            if label[index] in error_dict.keys():
                error_dict[label[index]] += 1   # 该类别分类错误的个数加１
                error_dict_record[label[index]].append(logit)   # 记录错误的结果
            else:
                error_dict[label[index]] = 1
                error_dict_record[label[index]] = [logit]
    acc = (1.0 * error_count) / (1.0 * len(label))
    if show:
        for key in error_dict.keys():
            print 'label is %d, error number is %d' % (key, error_dict[key])
            print 'error record　is ', error_dict_record[key]
    save_error_path = '/home/give/Documents/dataset/BOT_Game/val/error'
    saved_files = os.listdir(save_error_path)
    if images_name is not None:
        for image_name in error_names:
            if os.path.basename(image_name) in saved_files:
                print image_name, ' exists'
            else:
                print image_name, ' copy to ', os.path.join(save_error_path, os.path.basename(image_name))
                shutil.copy(
                    image_name,
                    os.path.join(save_error_path, os.path.basename(image_name))
                )
    return error_dict, error_dict_record, acc


# shuffle image and label
def shuffle_image_label(image, label):
    image = np.asarray(image)
    label = np.asarray(label)
    indexs = range(len(image))
    np.random.shuffle(indexs)
    image = image[indexs]
    label = label[indexs]
    return image, label


def resize_images(images, new_size):
    resized_images = []
    for image in images:
        img = Image.fromarray(image)
        img = img.resize(new_size)
        resized_images.append(
            np.array(img)
        )
    return resized_images


def calculate_accuracy_regression(logits, scores, diff_range):
    right_number = 0
    for index, score in enumerate(scores):
        logit = logits[index]
        diff_score = np.fabs(score-logit)
        if not (diff_score <= diff_range):
            print 'error, score is %g, logit is %g' % (score, logit)
        else:
            right_number += 1
    return (1.0 * right_number) / (1.0 * len(logits))


def calculate_tp(logits, labels):
    count = 0
    for index, logit in enumerate(logits):
        if logit == labels[index] and logit == 1:
            count += 1
    return count


def calculate_recall(logits, labels):
    tp = calculate_tp(logits=logits, labels=labels)
    recall = (tp * 1.0) / (np.sum(labels == 1) * 1.0)
    return recall


def calculate_precision(logits, labels):
    tp = calculate_tp(logits=logits, labels=labels)
    precision = (tp * 1.0) / (np.sum(logits == 1) * 1.0)
    return precision


def get_game_evaluate(logits, labels, argmax=None):
    logits = np.array(logits)
    labels = np.array(labels)
    if argmax is not None:
        logits = np.argmax(logits, argmax)
        labels = np.argmax(labels, argmax)
    recall = calculate_recall(logits=logits, labels=labels)
    precision = calculate_precision(logits=logits, labels=labels)
    f1_score = (2*precision*recall) / (precision + recall)
    return recall, precision, f1_score

'''
    计算每一个类别的准确率
'''
def calculate_accuracy(logits, labels, num_category=2):
    accuracy = []
    for i in range(num_category):
        flag = np.logical_and(
            logits == i,
            labels == i
        )
        accuracy.append(
            (1.0 * np.sum(flag)) / (1.0 * np.sum(labels == i))
        )
    return accuracy

def write_result(predicted_labels, predicted_names, save_path):
    filed = open(save_path, 'w')
    for index, label in enumerate(predicted_labels):
        name = predicted_names[index]
        if label:
            cur_str = name + ' P\n'
        else:
            cur_str = name + ' N\n'
        filed.write(cur_str)
    filed.close()

if __name__ == '__main__':
    logits = np.array([0, 1, 1, 1])
    labels = np.array([0, 1, 1, 0])
    print calculate_accuracy(logits, labels)