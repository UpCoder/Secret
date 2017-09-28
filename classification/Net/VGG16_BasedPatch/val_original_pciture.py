# -*- coding: utf-8 -*-
import tensorflow as tf
from trained_vgg16 import vgg16
from classification.dataset.DataSet import DataSet
import numpy as np
import gc
from classification.dataset.Config import Config as config
from Config import Config as net_config
from tools.Tools import resize_images, calculate_acc_error, get_game_evaluate
from tools.tools import calculate_loss, calculate_accuracy, save_weights
from tools.image_operations import calu_average_train_set


class val:
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

    def one_hot_encoding(self, labels):
        nb_classes = 2
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
        # avg_image = calu_average_train_set(config.TRAIN_ORIGINAL_IMAGE_DIR, [net_config.IMAGE_W, net_config.IMAGE_H])
        y = self.vgg.fcs_output
        # 计算准确率
        accuracy_tensor = calculate_accuracy(logits=y, labels=y_)
        saver = tf.train.Saver()
        labels = []
        logits = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if self.load_model_path:
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
                # cur_images -= avg_image
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
            recall, precision, f1_score = get_game_evaluate(
                logits=logits,
                labels=labels,
                argmax=None
            )
            print 'recall is %g \n presion is %g\nf1_score is %g\n' % (recall, precision, f1_score)
            # _, _, acc = calculate_acc_error(
            #     logits,
            #     labels,
            #     images_name=image_names
            # )
            # print 'accuracy is %g' % (1-acc)

if __name__ == '__main__':
    my_train = val(
        load_model_path='/home/give/PycharmProjects/StomachCanner/classification/Net/VGG16_BasedPatch/model/0.8_0.2_2_original_data/'
    )
    my_train.start_train()
