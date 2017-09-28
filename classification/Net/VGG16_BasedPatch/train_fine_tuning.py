# -*- coding: utf-8 -*-
import tensorflow as tf
from trained_vgg16 import vgg16
from classification.dataset.DataSet import DataSet
import numpy as np
import gc
from classification.dataset.Config import Config as config
from Config import Config as net_config
from tools.tools import calculate_loss, calculate_accuracy, save_weights
from tools.Tools import calculate_acc_error, resize_images
import os


class train:
    def __init__(self, load_model_path, save_model_path):
        self.load_model = load_model_path
        self.save_model_path = save_model_path
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
        merged = tf.summary.merge_all()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            log_path = './log/' + str(self.up_threshold) + '_' + str(self.down_threhold) + '/train'
            val_log_path = './log/' + str(self.up_threshold) + '_' + str(self.down_threhold) + '/val'
            if self.load_model:
                saver.restore(
                    sess,
                    self.load_model
                )
                print 'load model succssful', self.load_model
            writer = tf.summary.FileWriter(log_path, tf.get_default_graph())
            val_writer = tf.summary.FileWriter(val_log_path, tf.get_default_graph())
            for i in range(self.iterator_number):
                if i % 500 == 0 and i != 0:
                    print 'model save successful'
                    saver.save(
                        sess,
                        self.save_model_path
                    )
                train_images, labels = self.dataset.get_next_train_batch(net_config.TRAIN_BATCH_SIZE,
                                                                         net_config.TRAIN_BATCH_DISTRIBUTION)
                # print np.shape(train_images)
                # train_images = resize_images(
                #     train_images,
                #     [net_config.IMAGE_W, net_config.IMAGE_H]
                # )
                labels = self.one_hot_encoding(labels)
                feed_dict = {
                    self.vgg.imgs: train_images,
                    y_: labels
                }
                # if i%10 == 0:
                #     print labels
                _, loss_value, accuracy_value, summary, y_value, global_step_value = sess.run(
                    [train_op, loss, accuracy_tensor, merged, y, global_step],
                    feed_dict=feed_dict
                )
                # if i%10 == 0:
                #     print y_value
                writer.add_summary(summary, global_step_value)
                if (i % 10) == 0 and i != 0:
                    val_images, labels = self.dataset.get_next_val_batch(net_config.TRAIN_BATCH_SIZE,
                                                                         net_config.TRAIN_BATCH_DISTRIBUTION)
                    print 'val image shape is ', np.shape(val_images)
                    # print labels
                    # val_images = resize_images(
                    #     val_images,
                    #     [net_config.IMAGE_W, net_config.IMAGE_H]
                    # )
                    labels = self.one_hot_encoding(labels)
                    feed_dict = {
                        self.vgg.imgs: val_images,
                        y_: labels
                    }
                    global_step_value, val_loss, val_accuracy, summary, logits = sess.run(
                        [global_step, loss, accuracy_tensor, merged, y],
                        feed_dict=feed_dict
                    )
                    # print 'will save, accuracy is %g' % val_accuracy
                    # print np.shape(y_value)
                    calculate_acc_error(
                        logits=np.argmax(logits, 1),
                        label=np.argmax(labels, 1)
                    )
                    val_writer.add_summary(summary, global_step_value)
                    print '-' * 15, 'step is %d, train loss value is %g, accuracy is %g, val loss is %g, val accuracy is %g' % \
                                    (global_step_value, loss_value, accuracy_value, val_loss, val_accuracy), '-' * 15
                del train_images, labels
                gc.collect()
        writer.close()
        val_writer.close()


if __name__ == '__main__':
    my_train = train(
        load_model_path = '/home/give/PycharmProjects/StomachCanner/classification/Net/VGG16_BasedPatch/model/0.8_0.2_2_original_data/',
        save_model_path = '/home/give/PycharmProjects/StomachCanner/classification/Net/VGG16_BasedPatch/model/0.8_0.2_2_original_data/'
    )
    my_train.start_train()