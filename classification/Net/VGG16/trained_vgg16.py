# -*- coding: utf-8 -*-
########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
import numpy as np
from tools.tools import do_conv, pool, FC_layer, batch_norm, load_with_skip
from Config import Config

class vgg16:
    def __init__(self, imgs, weights=None, sess=None, skip_layers=['fc8']):
        self.imgs = imgs
        self.trainable = {
            'conv1_1': True,
            'conv1_2': True,
            'conv2_1': True,
            'conv2_2': True,
            'conv3_1': True,
            'conv3_2': True,
            'conv3_3': True,
            'conv4_1': True,
            'conv4_2': True,
            'conv4_3': True,
            'conv5_1': True,
            'conv5_2': True,
            'conv5_3': True,
            'fc6': True,
            'fc7': True,
            'fc8': True
        }
        self.layers_name = [
            'conv1_1',
            'conv1_2',
            'conv2_1',
            'conv2_2',
            'conv3_1',
            'conv3_2',
            'conv3_3',
            'conv4_1',
            'conv4_2',
            'conv4_3',
            'conv5_1',
            'conv5_2',
            'conv5_3',
            'fc6',
            'fc7',
            'fc8'
        ]
        self.classesnumber = 2
        # self.regularizer = None
        self.regularizer = tf.contrib.layers.l2_regularizer(Config.REGULARIZTION_RATE)
        self.convlayers()
        self.fc_layers()
        if weights is not None and sess is not None:
            load_with_skip(weights, sess, skip_layers)

    def convlayers(self):
        self.parameters = []

        # zero-mean input
        # with tf.name_scope('preprocess') as scope:
        #     mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        #     images = self.imgs-mean
        # 对输入数据进行归一化
        images = batch_norm(self.imgs)
        #　images = self.imgs
        # images = self.imgs
        # conv1_1
        # do_conv(name, input_tensor, out_channel, ksize, stride=[1, 1, 1, 1], is_pretrain=True):
        self.conv1_1 = do_conv('conv1_1', images, 64, [3, 3], is_pretrain=self.trainable['conv1_1'])
        self.conv1_2 = do_conv('conv1_2', self.conv1_1, 64, [3, 3], is_pretrain=self.trainable['conv1_2'],
                               batch_normalization=False)
        self.pooling1 = pool('pooling1', self.conv1_2, is_max_pool=True)

        self.conv2_1 = do_conv('conv2_1', self.pooling1, 128, [3, 3], is_pretrain=self.trainable['conv2_1'])
        self.conv2_2 = do_conv('conv2_2', self.conv2_1, 128, [3, 3], is_pretrain=self.trainable['conv2_2'],
                               batch_normalization=False)

        self.pooling2 = pool('pooling2', self.conv2_2, is_max_pool=True)

        self.conv3_1 = do_conv('conv3_1', self.pooling2, 256, [3, 3], is_pretrain=self.trainable['conv3_1'])
        self.conv3_2 = do_conv('conv3_2', self.conv3_1, 256, [3, 3], is_pretrain=self.trainable['conv3_2'])
        self.conv3_3 = do_conv('conv3_3', self.conv3_2, 256, [3, 3], is_pretrain=self.trainable['conv3_3'],
                               batch_normalization=False)

        self.pooling3 = pool('pooing3', self.conv3_3, is_max_pool=True)

        self.conv4_1 = do_conv('conv4_1', self.pooling3, 512, [3, 3], is_pretrain=self.trainable['conv4_1'])
        self.conv4_2 = do_conv('conv4_2', self.conv4_1, 512, [3, 3], is_pretrain=self.trainable['conv4_2'])
        self.conv4_3 = do_conv('conv4_3', self.conv4_2, 512, [3, 3], is_pretrain=self.trainable['conv4_3'],
                               batch_normalization=False)

        self.pooling4 = pool('pooling4', self.conv4_3, is_max_pool=True)

        self.conv5_1 = do_conv('conv5_1', self.pooling4, 512, [3, 3], is_pretrain=self.trainable['conv5_1'])
        self.conv5_2 = do_conv('conv5_2', self.conv5_1, 512, [3, 3], is_pretrain=self.trainable['conv5_2'])
        self.conv5_3 = do_conv('conv5_3', self.conv5_2, 512, [3, 3], is_pretrain=self.trainable['conv5_3'],
                               batch_normalization=False)

        self.pooling5 = pool('pooling5', self.conv5_3, is_max_pool=True)

        self.convs_output = self.pooling5

    def fc_layers(self):
        # def FC_layer(layer_name, x, out_nodes):
        images = FC_layer('fc6', self.convs_output, 4096, regularizer=self.regularizer, dropout=True)
        images = batch_norm(images)
        images = FC_layer('fc7', images, 4096, regularizer=self.regularizer, dropout=True)
        images = batch_norm(images)
        self.feature = images
        images = FC_layer('fc8', images, self.classesnumber, regularizer=self.regularizer)

        self.fcs_output = images

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print i, k, np.shape(weights[k])
            sess.run(self.parameters[i].assign(weights[k]))