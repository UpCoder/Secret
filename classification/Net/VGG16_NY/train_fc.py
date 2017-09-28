# -*- coding: utf-8 -*-
import tensorflow as tf
from trained_vgg16 import vgg16
from classification.dataset.PatchDataSetNY import PatchDataSetNY
import numpy as np
import gc
from classification.dataset.Config import Config as config
from Config import Config as net_config
from tools.Tools import resize_images
from tools.tools import calculate_loss, calculate_accuracy, save_weights
from tools.image_operations import calu_average_train_set

class train:
    def __init__(self, load_model_path, save_model_path, load_params_path):
        self.load_model_path = load_model_path
        self.save_model_path = save_model_path
        self.load_params_path = load_params_path
        self.BATCH_SIZE = 50
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
        self.dataset = PatchDataSetNY()
        self.learning_rate = 1e-5
        self.iterator_number = int(1e+5)
        # self.params_path = '/home/give/PycharmProjects/StomachCanner/classification/Net/VGG16/vgg16.npy'
        self.params_path = self.load_params_path
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
        # 获取平均图像值
        avg_image = self.dataset.train_data.get_average_image()
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
        train_step = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate
        ).minimize(loss, global_step=global_step)
        with tf.control_dependencies([train_step, variable_averages_op]):
            train_op = tf.no_op(name='train')
        # 计算准确率
        accuracy_tensor = calculate_accuracy(logits=y, labels=y_)
        merged = tf.summary.merge_all()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            log_path = './logs/train'
            val_log_path = './logs/val'
            if self.load_model_path:
                saver.restore(sess, self.load_model_path)
            writer = tf.summary.FileWriter(log_path, tf.get_default_graph())
            val_writer = tf.summary.FileWriter(val_log_path, tf.get_default_graph())
            for i in range(self.iterator_number):
                train_images, labels = self.dataset.get_next_train_batch(self.BATCH_SIZE)
                train_images = np.asarray(train_images, np.float32)
                train_images -= avg_image

                labels = self.one_hot_encoding(labels)
                feed_dict = {
                    self.vgg.imgs: train_images,
                    y_: labels
                }
                _, loss_value, accuracy_value, summary, y_value, global_step_value = sess.run(
                    [train_op, loss, accuracy_tensor, merged, y, global_step],
                    feed_dict=feed_dict
                )
                if i % 500 == 0 and i != 0:
                    # 保存模型
                    print 'save model successful', self.save_model_path
                    saver.save(sess, self.save_model_path)
                writer.add_summary(summary, i)
                if (i % 40) == 0 and i != 0:
                    val_images, labels = self.dataset.get_next_val_batch(self.BATCH_SIZE)
                    val_images = np.asarray(val_images, np.float32)
                    val_images -= avg_image
                    labels = self.one_hot_encoding(labels)
                    feed_dict = {
                        self.vgg.imgs: val_images,
                        y_: labels
                    }
                    val_loss, val_accuracy, summary = sess.run(
                        [loss, accuracy_tensor, merged],
                        feed_dict=feed_dict
                    )
                    val_writer.add_summary(summary, i)
                    print '-'*15, 'global_step is %d, train loss value is %g, accuracy is %g, val loss is %g, val accuracy is %g' % \
                                  (global_step_value, loss_value, accuracy_value, val_loss, val_accuracy), '-'*15
                del train_images, labels
                gc.collect()
        writer.close()
        val_writer.close()

if __name__ == '__main__':
    my_train = train(
        # load_model_path='/home/give/PycharmProjects/StomachCanner/classification/Net/VGG16_NY/models/',
        load_model_path=None,
        save_model_path='/home/give/PycharmProjects/StomachCanner/classification/Net/VGG16_NY/models/',
        # load_params_path=None
        load_params_path='/home/give/PycharmProjects/StomachCanner/classification/Net/VGG16_NY/vgg16.npy'
    )
    my_train.start_train()