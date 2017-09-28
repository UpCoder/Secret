class Config:
    NEED_MUL = True
    MODEL_SAVE_PATH = '/home/give/PycharmProjects/StomachCanner/classification/Net/model/'
    OUTPUT_NODE = 2
    CONV_LAYERS_CONFIG = {
        'conv1_1': {
            'deep': 32,
            'size': 3,
            'dropout': True,
            'pooling': {
                'exists': False,
                'name': 'pooling1'
            }
        },
        'conv2_1': {
            'deep': 32,
            'size': 3,
            'dropout': True,
            'pooling': {
                'exists': True,
                'name': 'pooling2'
            }
        },
        'conv3_1': {
            'deep': 64,
            'size': 3,
            'dropout': True,
            'pooling': {
                'exists': False,
                'name': 'pooling3'
            }
        },
        'conv4_1': {
            'deep': 64,
            'size': 3,
            'dropout': True,
            'pooling': {
                'exists': True,
                'name': 'pooling4'
            }
        },
        'conv5_1': {
            'deep': 128,
            'size': 3,
            'dropout': True,
            'pooling': {
                'exists': False,
                'name': 'pooling5'
            }
        },
        'conv6_1': {
            'deep': 128,
            'size': 3,
            'dropout': True,
            'pooling': {
                'exists': True,
                'name': 'pooling6'
            }
        },
        'conv7_1': {
            'deep': 256,
            'size': 3,
            'dropout': True,
            'pooling': {
                'exists': True,
                'name': 'pooling7'
            }
        }
    }

    FC_LAYERS_CONFIG = {
        'fc1': {
            'size': 2048,
            'regularizer': True,
            'batch_norm': True
        },
        'fc2': {
            'size': 1024,
            'regularizer': True,
            'batch_norm': True
        },
        'fc3': {
            'size': OUTPUT_NODE,
            'regularizer': True,
            'batch_norm': False
        }
    }

    TRAIN_BATCH_SIZE = 40
    TRAIN_BATCH_DISTRIBUTION = [
        20,    # cancer number
        20,    # non-cancer number
    ]

    IMAGE_W = 256
    IMAGE_H = 256
    IMAGE_CHANNEL = 3

    REGULARIZTION_RATE = 1.0

    LEARNING_RATE = 1e-3

    DROP_OUT = True

    ITERATOE_NUMBER = int(1e+4)

    TRAIN_LOG_DIR = '/home/give/PycharmProjects/StomachCanner/classification/Net/logs/train'
    VAL_LOG_DIR = '/home/give/PycharmProjects/StomachCanner/classification/Net/logs/val'
