import os
from PIL import Image
import numpy as np
import tensorflow as tf
from classification.Net.ResNet.image_processing import image_preprocessing

def read_images(image_dir):
    names = os.listdir(image_dir)
    pathes = [os.path.join(image_dir, name) for name in names]
    for path in pathes:
        image = Image.open(path)
        print path, ' shape: ', np.shape(image)
def tensorflow_read(image_dir):
    names = os.listdir(image_dir)
    pathes = [os.path.join(image_dir, name) for name in names]
    path_tensor = tf.placeholder(
        tf.string
    )
    image_buffer = tf.read_file(path_tensor)
    bbox = []
    image = image_preprocessing(
        image_buffer,
        bbox=bbox,
        train=True,
        thread_id=0
    )
    print tf.shape(image)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for path in pathes:
            print path
            image_value = sess.run(image, {
                path_tensor:path
            })
            print np.shape(image_value)
if __name__ == '__main__':
    tensorflow_read('/home/give/Documents/dataset/BOT_Game/patches/method4/train/negative')