"""Contains *modified* definition for the original form of Residual Networks.
"""
import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_utils
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_v2

resnet_arg_scope = resnet_utils.resnet_arg_scope
slim = tf.contrib.slim


@slim.add_arg_scope
def resnet_v1_50(inputs,
                 config,
                 is_training=True,
                 scope='resnet_v1_50'):
    """Modified ResNet-50 model."""
    # Note : The base_depth was reduced to be able to fit into GPU memory
    blocks = [
        resnet_v1.resnet_v1_block('block1',
                                  base_depth=config.block1_depth,
                                  num_units=config.block1_units,
                                  stride=config.block1_stride),
        resnet_v1.resnet_v1_block('block2',
                                  base_depth=config.block2_depth,
                                  num_units=config.block2_units,
                                  stride=config.block2_stride),
        resnet_v1.resnet_v1_block('block3',
                                  base_depth=config.block3_depth,
                                  num_units=config.block3_units,
                                  stride=config.block3_stride),
        resnet_v1.resnet_v1_block('block4',
                                  base_depth=config.block4_depth,
                                  num_units=config.block4_units,
                                  stride=config.block4_stride),
    ]
    return resnet_v1.resnet_v1(inputs,
                               blocks,
                               is_training=is_training,
                               global_pool=False,
                               include_root_block=True,
                               scope=scope)


def resnet_v2_50(inputs,
                 config,
                 is_training=True,
                 scope='resnet_v2_50'):
    """Modified ResNet-50 model."""
    blocks = [
        resnet_v2.resnet_v2_block('block1',
                                  base_depth=config.block1_depth,
                                  num_units=config.block1_units,
                                  stride=config.block1_stride),
        resnet_v2.resnet_v2_block('block2',
                                  base_depth=config.block2_depth,
                                  num_units=config.block2_units,
                                  stride=config.block2_stride),
        resnet_v2.resnet_v2_block('block3',
                                  base_depth=config.block3_depth,
                                  num_units=config.block3_units,
                                  stride=config.block3_stride),
        resnet_v2.resnet_v2_block('block4',
                                  base_depth=config.block4_depth,
                                  num_units=config.block4_units,
                                  stride=config.block4_stride),
    ]
    return resnet_v2.resnet_v2(inputs,
                               blocks,
                               is_training=is_training,
                               global_pool=False,
                               include_root_block=True,
                               scope=scope)
