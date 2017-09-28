# -*- coding: utf-8 -*-
import tensorflow as tf
from tools.tools import batch_norm, do_conv, FC_layer, pool
from Config import Config


# 实现ＬｅＮｅｔ网络结构
def inference(input_tensor, regularizer=None):
    # do_conv(name, input_tensor, out_channel, ksize, stride=[1, 1, 1, 1], is_pretrain=True, dropout=False, regularizer=None):
    layers = list(Config.CONV_LAYERS_CONFIG.keys())
    layers.sort()
    for key in layers:
        print key
        layer_config = Config.CONV_LAYERS_CONFIG[key]
        input_tensor = do_conv(
            key,
            input_tensor,
            layer_config['deep'],
            [layer_config['size'], layer_config['size']],
            dropout=layer_config['dropout']

        )
        if layer_config['pooling']['exists']:
            pooling = pool(
                layer_config['pooling']['name'],
                input_tensor
            )
            input_tensor = pooling
        print input_tensor

    # FC_layer(layer_name, x, out_nodes, regularizer=None):
    for key in Config.FC_LAYERS_CONFIG:
        layer_config = Config.FC_LAYERS_CONFIG[key]
        input_tensor = FC_layer(
            key,
            input_tensor,
            layer_config['size'],
            regularizer
        )
        if layer_config['batch_norm']:
            input_tensor = batch_norm(input_tensor)
    return input_tensor


def test_unit():
    input_tensor = tf.placeholder(
        tf.float32,
        [
            20,
            54,
            54,
            1
        ]
    )
    y = inference(input_tensor)
    if y.get_shape().as_list() == [20, 5]:
        print 'Success'
        return
    print 'Error'


if __name__ == '__main__':
    test_unit()