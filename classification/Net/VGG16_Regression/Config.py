# -*- coding: utf-8 -*-


class Config:
    NEED_MUL = True
    MODEL_LOAD_PATH = '/home/give/PycharmProjects/StomachCanner/classification/Net/VGG16_Regression/model/'
    MODEL_SAVE_PATH = '/home/give/PycharmProjects/StomachCanner/classification/Net/VGG16_Regression/model/'
    OUTPUT_NODE = 1     # 因为我们这里做的是回归问题，所以这里我们直接将输入节点的个数设置为１
    TRAIN_BATCH_SIZE = 30
    TRAIN_BATCH_DISTRIBUTION = [
        10,    # cancer number
        10,    # non-cancer number
        10
    ]

    IMAGE_W = 224
    IMAGE_H = 224
    IMAGE_CHANNEL = 3

    REGULARIZTION_RATE = 1e-4

    LEARNING_RATE = 1e-3
    MOVEING_AVERAGE_DECAY = 0.9997
    DROP_OUT = True

    ITERATOE_NUMBER = int(1e+4)

