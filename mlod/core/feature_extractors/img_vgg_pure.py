"""Contains wrapper for VGG model definition to extract features from
RGB image input.

Usage:
    outputs, end_points = ImgResnet(inputs, layers_config)
"""

import tensorflow as tf

from mlod.core.feature_extractors import img_feature_extractor
from mlod.core.feature_extractors import vgg

slim = tf.contrib.slim


class ImgVggPure(img_feature_extractor.ImgFeatureExtractor):

    def build(self,
              inputs,
              input_pixel_size,
              is_training,
              scope='vgg_16'):
        """Resnet for BEV feature extraction

        Note: All the fully_connected layers have been transformed to conv2d
              layers and are implemented in the main model.

        Args:
            inputs: a tensor of size [batch_size, height, width, channels].
            input_pixel_size: size of the input (H x W)
            is_training: True for training, False fo validation/testing.
            scope: Optional scope for the variables.

        Returns:
            The net, a rank-4 tensor of size [batch, height_out, width_out,
                channels_out] and end_points dict.
        """
        with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
          end_points_collection = sc.name + '_end_points'
          # Collect outputs for conv2d, fully_connected and max_pool2d.
          with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                              outputs_collections=end_points_collection):
              net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
              net = slim.max_pool2d(net, [2, 2], scope='pool1')
              net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
              net = slim.max_pool2d(net, [2, 2], scope='pool2')
              net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
              net = slim.max_pool2d(net, [2, 2], scope='pool3')
              net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
              net = slim.max_pool2d(net, [2, 2], scope='pool4')
              net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')

        # Convert end_points_collection into a end_point dict.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        
        feature_maps_out = net

        return feature_maps_out, end_points
