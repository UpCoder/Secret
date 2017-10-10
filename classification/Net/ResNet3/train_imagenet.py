import skimage.io  # bug. need to import this before tensorflow
import skimage.transform  # bug. need to import this before tensorflow
from resnet_train import train
import tensorflow as tf
import time
import os
import sys
import re
from net_config import Net_Config as net_config
import numpy as np
from resnet import inference
from DataSetBase import DataSetBase as DataSet
from image_processing import image_preprocessing

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '/home/ryan/data/ILSVRC2012/ILSVRC2012_img_train',
                           'imagenet dir')


def file_list(data_dir):
    dir_txt = data_dir + ".txt"
    filenames = []
    with open(dir_txt, 'r') as f:
        for line in f:
            if line[0] == '.': continue
            line = line.rstrip()
            fn = os.path.join(data_dir, line)
            filenames.append(fn)
    return filenames


def distorted_inputs_unit(pathes, labeles, trainable, shuffle=True):
    filenames = pathes
    labels = labeles
    filename, label = tf.train.slice_input_producer([filenames, labels], shuffle=shuffle)
    num_process_threads = 4
    images_and_labels = []
    for thread_id in range(num_process_threads):
        image_buffer = tf.read_file(filename)
        bbox = []
        image = image_preprocessing(
            image_buffer,
            bbox=bbox,
            train=trainable,
            thread_id=thread_id
        )
        # image = tf.image.rgb_to_hsv(image)
        images_and_labels.append([image, label])
    batch_image, batch_label = tf.train.batch_join(
        images_and_labels,
        batch_size=FLAGS.batch_size,
        capacity=2*num_process_threads*FLAGS.batch_size
    )
    height = net_config.IMAGE_W
    width = net_config.IMAGE_H
    depth = 3

    images = tf.cast(batch_image, tf.float32)
    images = tf.reshape(images, shape=[FLAGS.batch_size, height, width, depth])

    return images, tf.reshape(batch_label, [FLAGS.batch_size])

def distorted_inputs():
    def generate_path_label(cur_dir, label):
        names = os.listdir(cur_dir)
        pathes = [os.path.join(cur_dir, name) for name in names]
        labeles = [label] * len(pathes)
        return pathes, labeles
    flag0_path = '/home/give/Documents/dataset/BOT_Game/patches/method5/train/cell-512'
    flag1_path = '/home/give/Documents/dataset/BOT_Game/patches/method5/train/other-512'
    flag2_path = '/home/give/Documents/dataset/BOT_Game/patches/method5/train/positive-512'
    flag0_pathes, flag0_labels = generate_path_label(flag0_path, 0)
    flag1_pathes, flag1_labels = generate_path_label(flag1_path, 1)
    flag2_pathes, flag2_labels = generate_path_label(flag2_path, 2)
    pathes = []
    pathes.extend(flag0_pathes)
    pathes.extend(flag1_pathes)
    pathes.extend(flag2_pathes)
    labeles = []
    labeles.extend(flag0_labels)
    labeles.extend(flag1_labels)
    labeles.extend(flag2_labels)
    return distorted_inputs_unit(pathes, labeles, True)


def main(_):
    [train_images, train_labels] = distorted_inputs()
    print train_images
    is_training = tf.placeholder('bool', [], name='is_training')
    images, labels = tf.cond(is_training,
                             lambda: (train_images, train_labels),
                             lambda: (train_images, train_labels))
    logits = inference(images,
                       num_classes=net_config.OUTPUT_NODE,
                       is_training=True,
                       bottleneck=False,)
    save_model_path = '/home/give/PycharmProjects/StomachCanner/classification/Net/ResNet3/models/method5'
    train(is_training, logits, images, labels, save_model_path=save_model_path)


if __name__ == '__main__':
    tf.app.run()
