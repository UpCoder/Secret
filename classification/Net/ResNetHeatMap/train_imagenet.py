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


def distorted_inputs_unit(dataset, trainable, shuffle=True):
    filenames = dataset.images_names
    labels = dataset.labels
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
    train_positive_path = '/home/give/Documents/dataset/BOT_Game/train/positive-hm/method5'
    train_negative_path = '/home/give/Documents/dataset/BOT_Game/train/negative-hm/method6'
    val_positive_path = '/home/give/Documents/dataset/BOT_Game/val/positive-hm/method5'
    val_negative_path = '/home/give/Documents/dataset/BOT_Game/val/negative-hm/method5'
    val_dataset = DataSet(
        positive_path=val_positive_path,
        negative_path=val_negative_path
    )
    train_dataset = DataSet(
        positive_path=train_positive_path,
        negative_path=train_negative_path
    )
    return train_dataset, val_dataset


def generate_next_batch(dataset, batch_size, epoch_num):
    from PIL import Image
    cur_epoch = 1
    file_names = dataset.images_names
    labels = dataset.labels
    from tools.Tools import shuffle_image_label
    file_names, labels = shuffle_image_label(file_names, labels)
    start = 0
    while True:
        end = start + batch_size
        if end > len(file_names):
            end = len(file_names)
            cur_epoch += 1
        image_batchs = np.array(
            [np.array(Image.open(path).resize([net_config.IMAGE_W, net_config.IMAGE_H])) for path in
             file_names[start: end]]
        )

        for index, image in enumerate(image_batchs):
            for j in range(net_config.IMAGE_CHANNEL):
                if np.max(image[:, :, j]) != 0:
                    image_batchs[index, :, :, j] = (1.0 * image[:, :, j]) / (np.max(image[:, :, j]))
        label_batchs = labels[start: end]
        if end == len(file_names):
            start = 0
        else:
            start = end
        if epoch_num is not None and cur_epoch > epoch_num:
            break
        yield image_batchs, label_batchs


def main(_):
    train_dataset, val_dataset = distorted_inputs()
    train_batch = generate_next_batch(train_dataset, net_config.BATCH_SIZE, None)
    val_batch = generate_next_batch(val_dataset, net_config.BATCH_SIZE, None)
    is_training = tf.placeholder('bool', [], name='is_training')
    image_tensor = tf.placeholder(
        tf.float32,
        [None, net_config.IMAGE_W, net_config.IMAGE_H, net_config.IMAGE_CHANNEL]
    )
    label_tensor = tf.placeholder(
        tf.int32,
        [None]
    )
    logits = inference(image_tensor,
                       num_classes=2,
                       is_training=True,
                       bottleneck=False,)
    save_model_path = '/home/give/PycharmProjects/StomachCanner/classification/Net/ResNetHeatMap/models/method5-512/1740.0'
    train(is_training, logits, image_tensor, label_tensor, train_batch, val_batch, save_model_path=save_model_path)


if __name__ == '__main__':
    tf.app.run()
