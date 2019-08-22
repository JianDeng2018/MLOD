"""Contains wrapper for inception model definition to extract features from
Bird's eye view input.

Usage:
    outputs, end_points = BevInception(inputs, layers_config)
"""

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

from mlod.core.feature_extractors import bev_feature_extractor

slim = tf.contrib.slim


class BevInception(bev_feature_extractor.BevFeatureExtractor):

    def build(self,
              inputs,
              input_pixel_size,
              is_training,
              scope='bev_inception'):
        """Inception for BEV feature extraction

        Args:
            inputs: a tensor of size [batch_size, height, width, channels].
            input_pixel_size: size of the input (H x W)
            is_training: True for training, False fo validation/testing.
            scope: Optional scope for the variables.

        Returns:
            The net, a rank-4 tensor of size [batch, height_out, width_out,
                channels_out] and end_points dict.
        """

        inception_config = self.config

        with tf.variable_scope(
                scope, 'bev_inception', [inputs]) as scope:
            with slim.arg_scope(
                    [slim.batch_norm, slim.dropout],
                    is_training=is_training):

                if inception_config.inception_v == 'inception_v1':
                    with slim.arg_scope(inception.inception_v1_arg_scope()):
                        net, end_points = inception.inception_v1_base(
                                inputs, scope=scope)

                elif inception_config.inception_v == 'inception_v2':
                    with slim.arg_scope(inception.inception_v2_arg_scope()):
                        net, end_points = inception.inception_v2_base(
                                inputs, scope=scope)

                elif inception_config.inception_v == 'inception_v3':
                    with slim.arg_scope(inception.inception_v3_arg_scope()):
                        net, end_points = inception.inception_v3_base(
                                inputs, scope=scope)
                else:
                    raise ValueError('Invalid Inception version {},'.
                                     format(inception_config.inception_v))

                with tf.variable_scope('upsampling'):
                    # This feature extractor downsamples the input by a factor
                    # of 32
                    downsampling_factor = 32
                    downsampled_shape = input_pixel_size / downsampling_factor

                    upsampled_shape = downsampled_shape * \
                        inception_config.upsampling_multiplier

                    feature_maps_out = tf.image.resize_bilinear(
                        net, upsampled_shape)

        return feature_maps_out, end_points
