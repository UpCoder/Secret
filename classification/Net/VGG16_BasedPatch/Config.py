class Config:
    NEED_MUL = True
    MODEL_LOAD_PATH = '/home/give/PycharmProjects/StomachCanner/classification/Net/VGG16_BasedPatch/model/'
    MODEL_SAVE_PATH = '/home/give/PycharmProjects/StomachCanner/classification/Net/VGG16_BasedPatch/model/'
    OUTPUT_NODE = 2
    TRAIN_BATCH_SIZE = 40
    TRAIN_BATCH_DISTRIBUTION = [
        20,    # cancer number
        20,    # non-cancer number
    ]

    IMAGE_W = 224
    IMAGE_H = 224
    IMAGE_CHANNEL = 3

    REGULARIZTION_RATE = 1e-4

    LEARNING_RATE = 1e-7
    MOVEING_AVERAGE_DECAY = 0.9997
    DROP_OUT = True

    ITERATOE_NUMBER = int(1e+4)

    TRAIN_LOG_DIR = '/home/give/PycharmProjects/StomachCanner/classification/Net/VGG16_BasedPatch/logs/train'
    VAL_LOG_DIR = '/home/give/PycharmProjects/StomachCanner/classification/Net/VGG16_BasedPatch/logs/val'
