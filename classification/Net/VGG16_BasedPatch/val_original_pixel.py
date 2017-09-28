# -*- coding=utf-8 -*-
# 针对原图中的每个像素点进行分类，得到热力图，然后再根据生成的热力图来分类
import tensorflow as tf
from trained_vgg16 import vgg16
from classification.dataset.DataSet import DataSet
import numpy as np
import gc
from classification.dataset.Config import Config as config
from Config import Config as net_config
from tools.Tools import resize_images, calculate_acc_error
from tools.tools import calculate_loss, calculate_accuracy, save_weights
from tools.image_operations import calu_average_train_set, tiff_read, tiff_save


class val_orginal_pixel_wise:
    def __init__(self, load_model_path):
        self.dataset = DataSet(config)
        self.load_model_path = load_model_path
        self.BATCH_SIZE = 128
        self.sess = tf.Session()
        imgs = tf.placeholder(
            tf.float32,
            shape=[
                None,
                net_config.IMAGE_W,
                net_config.IMAGE_H,
                net_config.IMAGE_CHANNEL
            ]
        )
        self.dataset = DataSet(config)
        self.learning_rate = 1e-3
        self.iterator_number = int(1e+5)
        # self.params_path = '/home/give/PycharmProjects/StomachCanner/classification/Net/VGG16/vgg16.npy'
        self.params_path = None
        self.vgg = vgg16(imgs, self.params_path, self.sess, skip_layers=['fc8'])

    def generate_heat_map(self, original_path, save_path, patch_size=256, stride=16):
        y_tensor = self.vgg.fcs_output
        tiff_image = tiff_read(original_path)
        shape = list(np.shape(tiff_image))
        heat_map = np.zeros(
            [(shape[0] - net_config.IMAGE_W)/stride, (shape[1] - net_config.IMAGE_H)/stride],
            dtype=np.uint8
        )
        patches = []
        for x in range(patch_size / 2, shape[0] - patch_size / 2, stride):
            for y in range(patch_size / 2, shape[1] - patch_size / 2, stride):
                cur_patch = tiff_image[x - patch_size / 2:x + patch_size / 2, y - patch_size / 2: y + patch_size / 2, :]
                patches.append(cur_patch)
        start_index = 0
        batch_size = 40
        logits = []
        saver = tf.train.Saver()
        print 'len is ', len(patches)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, self.load_model_path)
            while start_index < len(patches):
                end_index = start_index + batch_size
                if end_index > len(patches):
                    end_index = len(patches)
                print 'end_index is ', end_index
                batch_data = patches[start_index:end_index]
                # print np.shape(batch_data)
                batch_data = resize_images(
                    batch_data,
                    [net_config.IMAGE_W, net_config.IMAGE_H]
                )
                # print np.shape(batch_data)
                # print self.vgg.imgs
                # print y_tensor
                cur_logits = sess.run(
                    y_tensor,
                    feed_dict={
                        self.vgg.imgs: batch_data
                    }
                )
                cur_logits = np.argmax(cur_logits, 1)
                logits.extend(cur_logits)
                start_index = end_index
        count_index = 0
        x_index = 0
        for x in range(patch_size / 2, shape[0] - patch_size / 2, stride):
            y_index = 0
            for y in range(patch_size / 2, shape[1] - patch_size / 2, stride):
                heat_map[x_index, y_index] = logits[count_index]*255
                count_index += 1
                y_index += 1
            x_index += 1
        tiff_save(heat_map, save_path)

    def one_hot_encoding(self, labels):
        nb_classes = net_config.OUTPUT_NODE
        targets = np.array([labels]).reshape(-1)
        one_hot_targets = np.eye(nb_classes)[targets]
        return one_hot_targets

    def start_train(self):
        y_ = tf.placeholder(
            tf.float32,
            [
                None,
                2
            ]
        )
        avg_image = calu_average_train_set(config.TRAIN_ORIGINAL_IMAGE_DIR, [net_config.IMAGE_W, net_config.IMAGE_H])
        y = self.vgg.fcs_output
        # 计算准确率
        accuracy_tensor = calculate_accuracy(logits=y, labels=y_)
        saver = tf.train.Saver()
        labels = []
        logits = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, self.load_model_path)
            image_names = self.dataset.get_val_merged_image_name()
            while self.dataset.get_val_has_next():
                cur_images, cur_labels = self.dataset.get_val_merged_next_batch(net_config.TRAIN_BATCH_SIZE)
                labels.extend(cur_labels)
                cur_images = resize_images(
                    cur_images,
                    [net_config.IMAGE_W, net_config.IMAGE_H]
                )
                cur_images = np.asarray(cur_images, np.float32)
                cur_images -= avg_image
                cur_labels = self.one_hot_encoding(cur_labels)
                feed_dict = {
                    self.vgg.imgs: cur_images,
                    y_: cur_labels
                }
                y_value, accuracy_value = sess.run(
                    [y, accuracy_tensor],
                    feed_dict=feed_dict
                )
                logits.extend(np.argmax(y_value, 1))
                print 'accuracy is ', accuracy_value
                del cur_images
                gc.collect()
            _, _, acc = calculate_acc_error(
                logits,
                labels,
                images_name=image_names
            )
            print 'accuracy is %g' % (1-acc)

if __name__ == '__main__':
    my_train = val_orginal_pixel_wise(
        load_model_path='/home/give/Documents/params/0.8_0.2_2/'
    )
    my_train.generate_heat_map(
        '/home/give/Documents/dataset/BOT_Game/train/negative/normal1.ndpi.16.5702_35104.2048x2048.tiff',
        '/home/give/Documents/dataset/BOT_Game/heat_map/train/negative/normal1.ndpi.16.5702_35104.2048x2048.tiff'
    )
