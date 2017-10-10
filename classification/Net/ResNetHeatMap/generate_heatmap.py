# # -*- coding=utf-8 -*-
# # 根据我们训练好的模型来生成概率图模型
# from tools.image_operations import extract_patchs_return
# import tensorflow as tf
# from net_config import Net_Config as net_config
# from resnet import inference
# import numpy as np
# import sys
# import math
# from PIL import Image
# import os
# MOMENTUM = 0.9
#
# FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string('train_dir', '/tmp/resnet_train',
#                            """Directory where to write event logs """
#                            """and checkpoint.""")
# tf.app.flags.DEFINE_string('save_model_dir', './models', 'the path using to save model')
# tf.app.flags.DEFINE_float('learning_rate', 0.01, "learning rate.")
# tf.app.flags.DEFINE_integer('batch_size', net_config.BATCH_SIZE, "batch size")
# tf.app.flags.DEFINE_integer('max_steps', 500000, "max steps")
# tf.app.flags.DEFINE_boolean('resume', True,
#                             'resume from latest saved state')
# # def train(logits, label_value, image_pathes):
# #     from image_processing import image_preprocessing
# #     filenames = image_pathes
# #     labels = label_value
# #     filename, label = tf.train.slice_input_producer([filenames, labels], shuffle=True)
# #     num_process_threads = 4
# #     images_and_labels = []
# #     for thread_id in range(num_process_threads):
# #         image_buffer = tf.read_file(filename)
# #         bbox = []
# #         image = image_preprocessing(
# #             image_buffer,
# #             bbox=bbox,
# #             train=False,
# #             thread_id=thread_id
# #         )
# #         # image = tf.image.rgb_to_hsv(image)
# #         images_and_labels.append([image, label])
# #     batch_image, batch_label = tf.train.batch_join(
# #         images_and_labels,
# #         batch_size=FLAGS.batch_size,
# #         capacity=2 * num_process_threads * FLAGS.batch_size
# #     )
# #     height = net_config.IMAGE_W
# #     width = net_config.IMAGE_H
# #     depth = 3
# #
# #     images = tf.cast(batch_image, tf.float32)
# #     images = tf.reshape(images, shape=[FLAGS.batch_size, height, width, depth])
# #
# #
# #     print 'image shape is ', np.shape(images)
# #     logits = inference(images,
# #                        num_classes=2,
# #                        is_training=False,
# #                        bottleneck=False)
# #     global_step = tf.get_variable('global_step', [],
# #                                   initializer=tf.constant_initializer(0),
# #                                   trainable=False)
# #     val_step = tf.get_variable('val_step', [],
# #                                   initializer=tf.constant_initializer(0),
# #                                   trainable=False)
# #     predictions = tf.nn.softmax(logits)
# #
# #     saver = tf.train.Saver(tf.all_variables())
# #
# #     init = tf.global_variables_initializer()
# #
# #     sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
# #     sess.run(init)
# #     sess.run(tf.initialize_local_variables())
# #     tf.train.start_queue_runners(sess=sess)
# #     print images.eval(session=sess)
# #     if FLAGS.resume:
# #         latest = tf.train.latest_checkpoint('/home/give/PycharmProjects/StomachCanner/classification/Net/ResNet/models/instance/5500.0/')
# #         if not latest:
# #             print "No checkpoint to continue from in", FLAGS.train_dir
# #             sys.exit(1)
# #         print "resume", latest
# #         saver.restore(sess, latest)
# #
# #     is_training = tf.placeholder('bool', [], name='is_training')
# #     predictions_values = sess.run(
# #         [predictions],
# #         {
# #             is_training: False
# #         })
# #     print predictions_values
# #     predictions_values = np.argmax(predictions_values, axis=1)
# #     print predictions_values
# '''
#     根据测试集的heating map得到分类结果
# '''
# def get_classification_result(image_dir):
#     names = os.listdir(image_dir)
#     image_pathes = [os.path.join(image_dir, name) for name in names]
#     filenames = image_pathes
#     print image_pathes
#     [filename] = tf.train.slice_input_producer([filenames], shuffle=False, num_epochs=1)
#     num_process_threads = 4
#     images_and_labels = []
#     from image_processing import image_preprocessing
#     for thread_id in range(num_process_threads):
#         image_buffer = tf.read_file(filename)
#         bbox = []
#         image = image_preprocessing(
#             image_buffer,
#             bbox=bbox,
#             train=False,
#             thread_id=thread_id
#         )
#         # image = tf.image.rgb_to_hsv(image)
#         images_and_labels.append([image])
#     batch_image = tf.train.batch_join(
#         images_and_labels,
#         batch_size=1,
#         capacity=2 * num_process_threads * FLAGS.batch_size
#     )
#     height = net_config.IMAGE_W
#     width = net_config.IMAGE_H
#     depth = 3
#
#     images = tf.cast(batch_image, tf.float32)
#     images = tf.reshape(images, shape=[1, height, width, depth])
#     print images
#
#     logits = inference(images,
#                        num_classes=2,
#                        is_training=True,
#                        bottleneck=False, )
#
#     saver = tf.train.Saver(tf.all_variables())
#
#     init = tf.global_variables_initializer()
#
#     sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
#     sess.run(init)
#     sess.run(tf.initialize_local_variables())
#     tf.train.start_queue_runners(sess=sess)
#
#     latest = tf.train.latest_checkpoint(
#         '/home/give/PycharmProjects/StomachCanner/classification/Net/ResNetHeatMap/models/method5-512')
#     if not latest:
#         print "No checkpoint to continue from in", FLAGS.train_dir
#         sys.exit(1)
#     print "resume", latest
#     saver.restore(sess, latest)
#     predictions = tf.nn.softmax(logits)
#     predictions_label = tf.argmax(predictions, axis=1)
#     print predictions_label
#     while True:
#         prediction_value = sess.run(predictions_label)
#         print prediction_value
#     return images
#
# '''
#     针对heating map分类，返回每个heating map的分类结果
# '''
# def generate_prediction(patches):
#
#     probability = tf.nn.softmax(logits)
#     if FLAGS.resume:
#         latest = tf.train.latest_checkpoint(
#             '/home/give/PycharmProjects/StomachCanner/classification/Net/ResNetHeatMap/models/method5-512')
#         if not latest:
#             print "No checkpoint to continue from in", FLAGS.train_dir
#             sys.exit(1)
#         print "resume", latest
#         saver.restore(sess, latest)
#     probability_values = []
#     start = 0
#     batch_size = 512
#     while start < len(patches):
#         end = start + batch_size
#         if end >= len(patches):
#             end = len(patches)
#         cur_patches = patches[start:end]
#         probability_value = sess.run(
#             probability,
#             {
#                 img_tensor: cur_patches
#             }
#         )
#         # print probability_value
#         probability_values.extend(probability_value)
#         # print 'logits value shape is ', np.shape(probability_value)
#         start = end
#     probability_values = np.asarray(probability_values, np.float32)
#     return np.argmax(probability_values, axis=1)
#
# '''
#     加载已知模型，计算一个tiff文件的heat map
#     :param tiff_path 一个tiff文件的path
#     :param save_path 保存heat map 的路径 如果是None的话，则show
# '''
# def generate_heatmap(tiff_path, save_path):
#     if os.path.exists(save_path):
#         print 'Exists'
#         return
#     patches = extract_patchs_return(
#         tiff_path=tiff_path,
#         mask_dir=None,
#         occupy_rate=None,
#         stride=16,
#         patch_size=256
#     )
#     patches = np.asarray(patches, np.float32)
#     for index, patch in enumerate(patches):
#         patch = np.asarray(patch, np.float32)
#         patch = patch * (1.0 / np.max(patch))
#         patches[index] = patch
#     probability_value = generate_prediction(patches)
#     print np.max(probability_value), np.min(probability_value)
#     print probability_value
#     w = int(math.sqrt(len(probability_value)))
#     probability_img = Image.fromarray(np.asarray(np.reshape(probability_value, [w, w]) * 255, np.uint8))
#     if save_path is not None:
#         probability_img.save(save_path)
#     else:
#         probability_img.show()
#
#
# '''
#     加载已知模型，计算一个文件夹下面所有tiff文件的heat mapping
#     :param tiff_path 一个tiff文件的path
#     :param save_path 保存heat map 的路径 如果是None的话，则show
# '''
# def generate_heatmap_one_floder(tiff_dir, save_dir):
#     names = os.listdir(tiff_dir)
#     tiff_paths = [os.path.join(tiff_dir, name) for name in names]
#     for index, tiff_path in enumerate(tiff_paths):
#         name = names[index].split('.tiff')[0]
#         generate_heatmap(tiff_path, os.path.join(save_dir, name+'.png'))
#
#
# '''
#      加载已知模型，计算多个文件夹下面所有tiff文件的heat mapping
#      :param tiff_dirs 多个文件夹的路径
#      :param save_dirs 对上面参数对应的保存的路径
#
# '''
# def generate_heatmap_multi_floder(tiff_dirs, save_dirs):
#     for tiff_dir_index, tiff_dir in enumerate(tiff_dirs):
#         save_dir = save_dirs[tiff_dir_index]
#         names = os.listdir(tiff_dir)
#         tiff_paths = [os.path.join(tiff_dir, name) for name in names]
#         for index, tiff_path in enumerate(tiff_paths):
#             name = names[index].split('.tiff')[0]
#             generate_heatmap(tiff_path, os.path.join(save_dir, name+'.png'))
#
# if __name__ == '__main__':
#     # generate_heatmap_multi_floder(
#     #     tiff_dirs=[
#     #         '/home/give/Documents/dataset/BOT_Game/val/positive',
#     #         '/home/give/Documents/dataset/BOT_Game/val/negative'
#     #         # '/home/give/Documents/dataset/BOT_Game/0-testdataset'
#     #     ],
#     #     save_dirs=[
#     #         # '/home/give/Documents/dataset/BOT_Game/0-testdataset-hm'
#     #         '/home/give/Documents/dataset/BOT_Game/val/positive-hm',
#     #         '/home/give/Documents/dataset/BOT_Game/val/negative-hm'
#     #     ]
#     # )
#     # from tools.image_operations import read_images
#     # image_dir = '/home/give/Documents/dataset/BOT_Game/train/positive-test'
#     # names = os.listdir(image_dir)
#     # pathes = [os.path.join(image_dir, name) for name in names]
#     # patches = read_images('/home/give/Documents/dataset/BOT_Game/train/positive-test')
#     # for index, patch in enumerate(patches):
#     #     patch = np.asarray(patch, np.float32)
#     #     patch = patch * (1.0 / np.max(patch))
#     #     patches[index] = patch
#     # print 'patch shape is ', np.shape(patches)
#     # predicted = generate_prediction(patches)
#     # print np.max(predicted), np.min(predicted)
#     # print predicted
#     # train(None, [0]*len(pathes), pathes)
#     # get_classification_result('/home/give/Documents/dataset/BOT_Game/0-testdataset-hm/method5')
#     get_classification_result('/home/give/Documents/dataset/BOT_Game/train/positive-hm/method5')

import tensorflow as tf
import os
from net_config import Net_Config as net_config
from resnet import inference
from DataSetBase import DataSetBase as DataSet
from image_processing import image_preprocessing
from resnet_val import val
import numpy as np
from PIL import Image

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
        batch_size=net_config.BATCH_SIZE,
        capacity=2*num_process_threads*net_config.BATCH_SIZE
    )
    height = net_config.IMAGE_W
    width = net_config.IMAGE_H
    depth = 3

    images = tf.cast(batch_image, tf.float32)
    images = tf.reshape(images, shape=[net_config.BATCH_SIZE, height, width, depth])

    return images, tf.reshape(batch_label, [net_config.BATCH_SIZE])

def distorted_inputs():
    # data = load_data(FLAGS.data_dir)

    # filenames = [ d['filename'] for d in data ]
    # label_indexes = [ d['label_index'] for d in data ]
    # train_positive_path = '/home/give/Documents/dataset/BOT_Game/train/positive-png'
    # train_negative_path = '/home/give/Documents/dataset/BOT_Game/train/negative-copy'
    # val_positive_path = '/home/give/Documents/dataset/BOT_Game/val/positive-png'
    # val_negative_path = '/home/give/Documents/dataset/BOT_Game/val/negative-png'
    train_positive_path = '/home/give/Documents/dataset/BOT_Game/train/negative-hm/method6'
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
    return distorted_inputs_unit(train_dataset, False), distorted_inputs_unit(val_dataset, False)


def main(_):
    predict_dir = '/home/give/Documents/dataset/BOT_Game/train/positive-hm/method5'
    file_names = os.listdir(predict_dir)
    file_pathes = [os.path.join(predict_dir, file_name) for file_name in file_names]
    image_values = [np.array(Image.open(file_path).convert('RGB')) for file_path in file_pathes]
    image_values = np.asarray(image_values, np.float32)
    image_values = image_values[:net_config.BATCH_SIZE]
    new_image_values = []
    for index, image_value in enumerate(image_values):
        image_value = np.asarray(image_value, np.float32)
        image_value = image_value * (1.0 / np.max(image_value))
        image_value = np.asarray(image_value, np.float32)
        img = np.zeros([net_config.IMAGE_W, net_config.IMAGE_H, net_config.IMAGE_CHANNEL])
        for j in range(net_config.IMAGE_CHANNEL):
            img[:, :, j] = np.array(
                Image.fromarray(image_value[:, :, j]).resize([net_config.IMAGE_W, net_config.IMAGE_H])
            )
        new_image_values.append(np.array(img))
    image_values = np.array(new_image_values)
    image_tensor = tf.placeholder(
        tf.float32,
        [net_config.BATCH_SIZE, net_config.IMAGE_W, net_config.IMAGE_H, net_config.IMAGE_CHANNEL]
    )
    label_tensor = tf.placeholder(
        tf.int32,
        [net_config.BATCH_SIZE]
    )
    logits = inference(image_tensor,
                       num_classes=2,
                       is_training=True,
                       bottleneck=False,)
    save_model_path = '/home/give/PycharmProjects/StomachCanner/classification/Net/ResNetHeatMap/models/method5-512'
    print 'image_tensor is ', image_tensor
    print np.shape(image_values)
    val(image_tensor, logits, image_values, label_tensor, [0]*len(image_values), save_model_path=save_model_path)


if __name__ == '__main__':
    tf.app.run()

