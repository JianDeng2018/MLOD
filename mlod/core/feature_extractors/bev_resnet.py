"""Contains wrapper for Resnet model definition to extract features from
Bird's eye view input.

Usage:
    outputs, end_points = BevResnet(inputs, layers_config)
"""

import tensorflow as tf

from mlod.core.feature_extractors import bev_feature_extractor
from mlod.core.feature_extractors import resnet

slim = tf.contrib.slim


class BevResnet(bev_feature_extractor.BevFeatureExtractor):

    def build(self,
              inputs,
              input_pixel_size,
              is_training,
              scope='bev_resnet'):
        """Resnet for BEV feature extraction

        Args:
            inputs: a tensor of size [batch_size, height, width, channels].
            input_pixel_size: size of the input (H x W)
            is_training: True for training, False fo validation/testing.
            scope: Optional scope for the variables.

        Returns:
            The net, a rank-4 tensor of size [batch, height_out, width_out,
                channels_out] and end_points dict.
        """

        resnet_config = self.config
        with slim.arg_scope(resnet.resnet_arg_scope()):
            with tf.variable_scope(scope, 'bev_resnet'):
                if resnet_config.resnet_v == 'resnet_v1':
                    net, end_points = resnet.resnet_v1_50(
                        inputs, resnet_config, is_training=is_training)
                elif resnet_config.resnet_v == 'resnet_v2':
                    net, end_points = resnet.resnet_v2_50(
                        inputs, resnet_config, is_training=is_training)
                else:
                    raise ValueError('Invalid Resnet version {},'.
                                     format(resnet_config.resnet_v))

                with tf.variable_scope('upsampling'):
                    # This feature extractor downsamples the input by a factor
                    # of 32
                    downsampling_factor = 32
                    downsampled_shape = input_pixel_size / downsampling_factor

                    upsampled_shape = \
                        downsampled_shape * resnet_config.upsampling_multiplier

                    feature_maps_out = tf.image.resize_bilinear(
                        net, upsampled_shape)

        return feature_maps_out, end_points
