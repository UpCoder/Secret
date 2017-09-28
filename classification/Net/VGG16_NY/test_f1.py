# -*- coding: utf-8 -*-
import tensorflow as tf
from trained_vgg16 import vgg16
from classification.dataset.DataSet import DataSet
import numpy as np
import gc
from classification.dataset.Config import Config as config
from Config import Config as net_config
from tools.tools import calculate_loss, calculate_accuracy, save_weights
from tools.Tools import calculate_acc_error, resize_images, get_game_evaluate
from tools.image_operations import changed_shape
import os


class train:
    def __init__(self, load_model_path):
        self.load_model = load_model_path
        self.up_threshold = 0.8
        self.down_threhold = 0.2
        self.threshold = 0.7
        self.using_border = False
        self.dataset = DataSet(config)
        # self.dataset = PatchDataSet(config, self.threshold)
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
        self.learning_rate = 1e-3
        self.iterator_number = int(1e+5)
        if not self.load_model:
            model_load_path = '/home/give/PycharmProjects/StomachCanner/classification/Net/VGG16/vgg16.npy'
        else:
            model_load_path = None
        self.vgg = vgg16(imgs, model_load_path, self.sess)

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
                net_config.OUTPUT_NODE
            ]
        )
        y = self.vgg.fcs_output
        global_step = tf.Variable(0, trainable=False)
        variable_averages = tf.train.ExponentialMovingAverage(
            net_config.MOVEING_AVERAGE_DECAY,
            global_step
        )
        variable_averages_op = variable_averages.apply(
            tf.trainable_variables()
        )
        loss = calculate_loss(logits=y, labels=y_)
        tf.summary.scalar(
            'loss',
            loss
        )
        train_step = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate
        ).minimize(
            loss=loss,
            global_step=global_step
        )
        with tf.control_dependencies([train_step, variable_averages_op]):
            train_op = tf.no_op(name='train')
        # 计算准确率
        accuracy_tensor = calculate_accuracy(logits=y, labels=y_)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if self.load_model:
                saver.restore(
                    sess,
                    self.load_model
                )
                print 'load model succssful', self.load_model
            all_labels = []
            all_logits = []
            while True:
                val_images, labels, flag = self.dataset.get_val_merged_next_batch(net_config.TRAIN_BATCH_SIZE)
                if not flag:
                    break
                all_labels.extend(labels)
                labels = self.one_hot_encoding(labels)
                feed_dict = {
                    self.vgg.imgs: val_images,
                    y_: labels
                }
                global_step_value, val_loss, val_accuracy, logits = sess.run(
                    [global_step, loss, accuracy_tensor, y],
                    feed_dict=feed_dict
                )
                all_logits.extend(np.argmax(logits, 1))
                calculate_acc_error(
                    logits=np.argmax(logits, 1),
                    label=np.argmax(labels, 1)
                )
                print '-' * 15, 'val accuracy is %g' % \
                                (val_accuracy), '-' * 15
            recall, precision, f1_score = get_game_evaluate(
                logits=all_logits,
                labels=all_labels
            )
            print 'recall is %g, precision is %g, f1_score is %g' % (recall, precision, f1_score)


if __name__ == '__main__':
    my_train = train(
        load_model_path='/home/give/PycharmProjects/StomachCanner/classification/Net/VGG16_NY/models_orginal/'
    )
    my_train.start_train()