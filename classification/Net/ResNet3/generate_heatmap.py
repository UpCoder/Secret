# -*- coding=utf-8 -*-
# 根据我们训练好的模型来生成概率图模型
from tools.image_operations import extract_patchs_return
import tensorflow as tf
from net_config import Net_Config as net_config
from resnet import inference
import numpy as np
import sys
import math
from PIL import Image
import os
MOMENTUM = 0.9

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/tmp/resnet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('save_model_dir', './models', 'the path using to save model')
tf.app.flags.DEFINE_float('learning_rate', 0.01, "learning rate.")
tf.app.flags.DEFINE_integer('batch_size', net_config.BATCH_SIZE, "batch size")
tf.app.flags.DEFINE_integer('max_steps', 500000, "max steps")
tf.app.flags.DEFINE_boolean('resume', True,
                            'resume from latest saved state')
img_tensor = tf.placeholder(
    tf.float32,
    [
        None,
        net_config.IMAGE_W,
        net_config.IMAGE_H,
        net_config.IMAGE_CHANNEL
    ]
)
label_tensor = tf.placeholder(
    tf.int32,
    [None]
)
logits = inference(img_tensor,
                   num_classes=net_config.OUTPUT_NODE,
                   is_training=False,
                   bottleneck=False)

saver = tf.train.Saver(tf.all_variables())

init = tf.global_variables_initializer()

sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
sess.run(init)
sess.run(tf.initialize_local_variables())
tf.train.start_queue_runners(sess=sess)
# def train(logits, label_value, image_pathes):
#     from image_processing import image_preprocessing
#     filenames = image_pathes
#     labels = label_value
#     filename, label = tf.train.slice_input_producer([filenames, labels], shuffle=True)
#     num_process_threads = 4
#     images_and_labels = []
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
#         images_and_labels.append([image, label])
#     batch_image, batch_label = tf.train.batch_join(
#         images_and_labels,
#         batch_size=FLAGS.batch_size,
#         capacity=2 * num_process_threads * FLAGS.batch_size
#     )
#     height = net_config.IMAGE_W
#     width = net_config.IMAGE_H
#     depth = 3
#
#     images = tf.cast(batch_image, tf.float32)
#     images = tf.reshape(images, shape=[FLAGS.batch_size, height, width, depth])
#
#
#     print 'image shape is ', np.shape(images)
#     logits = inference(images,
#                        num_classes=2,
#                        is_training=False,
#                        bottleneck=False)
#     global_step = tf.get_variable('global_step', [],
#                                   initializer=tf.constant_initializer(0),
#                                   trainable=False)
#     val_step = tf.get_variable('val_step', [],
#                                   initializer=tf.constant_initializer(0),
#                                   trainable=False)
#     predictions = tf.nn.softmax(logits)
#
#     saver = tf.train.Saver(tf.all_variables())
#
#     init = tf.global_variables_initializer()
#
#     sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
#     sess.run(init)
#     sess.run(tf.initialize_local_variables())
#     tf.train.start_queue_runners(sess=sess)
#     print images.eval(session=sess)
#     if FLAGS.resume:
#         latest = tf.train.latest_checkpoint('/home/give/PycharmProjects/StomachCanner/classification/Net/ResNet/models/instance/5500.0/')
#         if not latest:
#             print "No checkpoint to continue from in", FLAGS.train_dir
#             sys.exit(1)
#         print "resume", latest
#         saver.restore(sess, latest)
#
#     is_training = tf.placeholder('bool', [], name='is_training')
#     predictions_values = sess.run(
#         [predictions],
#         {
#             is_training: False
#         })
#     print predictions_values
#     predictions_values = np.argmax(predictions_values, axis=1)
#     print predictions_values
'''
    针对每一个patch分类
'''
def generate_prediction(patches):
    probability = tf.nn.softmax(logits)
    if FLAGS.resume:
        latest = tf.train.latest_checkpoint(
            '/home/give/PycharmProjects/StomachCanner/classification/Net/ResNet3/models/method5/5000.0')
        if not latest:
            print "No checkpoint to continue from in", FLAGS.train_dir
            sys.exit(1)
        print "resume", latest
        saver.restore(sess, latest)
    probability_values = []
    start = 0
    batch_size = 512
    while start < len(patches):
        end = start + batch_size
        if end >= len(patches):
            end = len(patches)
        cur_patches = patches[start:end]
        probability_value = sess.run(
            probability,
            {
                img_tensor: cur_patches
            }
        )
        # print probability_value
        probability_values.extend(probability_value)
        # print 'logits value shape is ', np.shape(probability_value)
        start = end
    probability_values = np.asarray(probability_values, np.float32)
    w = int(math.sqrt(len(probability_values[:, 0])))
    res_image = np.zeros([w, w, 3], np.uint8)
    for i in range(net_config.OUTPUT_NODE):
        probability_value = probability_values[:, i]
        w = int(math.sqrt(len(probability_value)))
        probability_image =np.asarray(np.reshape(probability_value, [w, w])) * 255
        res_image[:, :, i] = probability_image
    return res_image

'''
    加载已知模型，计算一个tiff文件的heat map
    :param tiff_path 一个tiff文件的path
    :param save_path 保存heat map 的路径 如果是None的话，则show
'''
def generate_heatmap(tiff_path, save_path):
    if os.path.exists(save_path):
        print 'Exists', tiff_path
        return
    print tiff_path, ' will doing'
    patches = extract_patchs_return(
        tiff_path=tiff_path,
        mask_dir=None,
        occupy_rate=None,
        stride=16,
        patch_size=512
    )
    new_patches = []
    for patch in patches:
        img = Image.fromarray(patch)
        img = img.resize([net_config.IMAGE_W, net_config.IMAGE_H])
        new_patches.append(np.array(img))
    patches = new_patches
    patches = np.asarray(patches, np.float32)
    for index, patch in enumerate(patches):
        patch = np.asarray(patch, np.float32)
        patch = patch * (1.0 / np.max(patch))
        patches[index] = patch
    res_image = generate_prediction(patches)
    res_img = Image.fromarray(np.asarray(res_image, np.uint8))
    if save_path is not None:
        res_img.save(save_path)
    else:
        res_img.show()


'''
    加载已知模型，计算一个文件夹下面所有tiff文件的heat mapping
    :param tiff_path 一个tiff文件的path
    :param save_path 保存heat map 的路径 如果是None的话，则show
'''
def generate_heatmap_one_floder(tiff_dir, save_dir):
    names = os.listdir(tiff_dir)
    tiff_paths = [os.path.join(tiff_dir, name) for name in names]
    for index, tiff_path in enumerate(tiff_paths):
        name = names[index].split('.tiff')[0]
        generate_heatmap(tiff_path, os.path.join(save_dir, name+'.png'))


'''
     加载已知模型，计算多个文件夹下面所有tiff文件的heat mapping
     :param tiff_dirs 多个文件夹的路径
     :param save_dirs 对上面参数对应的保存的路径
     
'''
def generate_heatmap_multi_floder(tiff_dirs, save_dirs, process_num=1):
    # from multiprocessing import Process
    # def single_process(tiff_paths, names, process_id):
    #     print 'process id: ', process_id
    #     for index, tiff_path in enumerate(tiff_paths):
    #         name = names[index].split('.tiff')[0]
    #         generate_heatmap(tiff_path, os.path.join(save_dir, name+'.png'))
    # for tiff_dir_index, tiff_dir in enumerate(tiff_dirs):
    #     save_dir = save_dirs[tiff_dir_index]
    #     names = os.listdir(tiff_dir)
    #     tiff_paths = [os.path.join(tiff_dir, name) for name in names]
    #     pre_process_num = int(len(tiff_paths) / process_num + 1)
    #     start = 0
    #     processes = []
    #     for i in range(process_num):
    #         end = start + pre_process_num
    #         if end > len(tiff_paths):
    #             end = tiff_paths
    #         process = Process(target=single_process, args=[
    #             tiff_paths[start:end], names[start:end], i,
    #         ])
    #         process.start()
    #         processes.append(process)
    #     for process in processes:
    #         process.join()
    for tiff_dir_index, tiff_dir in enumerate(tiff_dirs):
        save_dir = save_dirs[tiff_dir_index]
        names = os.listdir(tiff_dir)
        tiff_paths = [os.path.join(tiff_dir, name) for name in names]
        for index, tiff_path in enumerate(tiff_paths):
            name = names[index].split('.tiff')[0]
            generate_heatmap(tiff_path, os.path.join(save_dir, name + '.png'))

if __name__ == '__main__':
    # generate_heatmap('/home/give/Documents/dataset/BOT_Game/val/positive/2017-06-10_19.46.43.ndpi.16.46032_19684.2048x2048.tiff',
    #                  '/home/give/Documents/dataset/BOT_Game/train/positive-hm/method5/2017-06-10_19.46.43.ndpi.16.46032_19684.2048x2048.png')
    generate_heatmap_multi_floder(
        tiff_dirs=[
            '/home/give/Documents/dataset/BOT_Game/train/negative',
            '/home/give/Documents/dataset/BOT_Game/train/positive',
            '/home/give/Documents/dataset/BOT_Game/val/negative',
            '/home/give/Documents/dataset/BOT_Game/val/positive',
            '/home/give/Documents/dataset/BOT_Game/0-testdataset'
        ],
        save_dirs=[
            '/home/give/Documents/dataset/BOT_Game/train/negative-hm/method5',
            '/home/give/Documents/dataset/BOT_Game/train/positive-hm/method5',
            '/home/give/Documents/dataset/BOT_Game/val/negative-hm/method5',
            '/home/give/Documents/dataset/BOT_Game/val/positive-hm/method5',
            '/home/give/Documents/dataset/BOT_Game/0-testdataset-hm/method5'
        ]
    )
    # from tools.image_operations import read_images
    # image_dir = '/home/give/Documents/dataset/BOT_Game/train/positive-test'
    # names = os.listdir(image_dir)
    # pathes = [os.path.join(image_dir, name) for name in names]
    # patches = read_images('/home/give/Documents/dataset/BOT_Game/train/positive-test')
    # for index, patch in enumerate(patches):
    #     patch = np.asarray(patch, np.float32)
    #     patch = patch * (1.0 / np.max(patch))
    #     patches[index] = patch
    # print 'patch shape is ', np.shape(patches)
    # predicted = generate_prediction(patches)
    # print np.max(predicted), np.min(predicted)
    # print predicted
    # train(None, [0]*len(pathes), pathes)