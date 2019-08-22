import tensorflow as tf

from mlod.core.feature_extractors import bev_feature_extractor

slim = tf.contrib.slim


class BevVggLfe(bev_feature_extractor.BevFeatureExtractor):
    """Contains modified VGG model definition to extract features from
    Bird's eye view input using pyramid features.
    """

    def vgg_arg_scope(self, weight_decay=0.0005):
        """Defines the VGG arg scope.

        Args:
          weight_decay: The l2 regularization coefficient.

        Returns:
          An arg_scope.
        """
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(
                                weight_decay),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
                return arg_sc

    def dilated_conv_block(self,inputs, out_channels, repeat_times, rate, scope_name):
        """
        implimentation of dilated convolution by using batch_to_space
        https://www.tensorflow.org/api_docs/python/tf/nn/atrous_conv2d
        """
        
        if rate == 1:
            outputs = slim.repeat(inputs,
                                  repeat_times,
                                  slim.conv2d,
                                  out_channels,
                                  [3, 3],
                                  scope=scope_name)
        elif rate > 1:
            pad = [[0,0],[0,0]]
            inputs_batch = tf.space_to_batch_nd(inputs, paddings=pad, block_shape=[rate,rate])
            outputs_batch = slim.repeat(inputs_batch,
                                  repeat_times,
                                  slim.conv2d,
                                  out_channels,
                                  [3, 3],
                                  scope=scope_name)

            outputs = tf.batch_to_space_nd(outputs_batch, crops=pad, block_shape=[rate,rate])

        return outputs

    def build(self,
              inputs,
              input_pixel_size,
              is_training,
              scope='bev_vgg_lfe'):
        """ Modified VGG for BEV feature extraction with pyramid features

        Args:
            inputs: a tensor of size [batch_size, height, width, channels].
            input_pixel_size: size of the input (H x W)
            is_training: True for training, False for validation/testing.
            scope: Optional scope for the variables.

        Returns:
            The last op containing the log predictions and end_points dict.
        """
        vgg_config = self.config

        with slim.arg_scope(self.vgg_arg_scope(
                weight_decay=vgg_config.l2_weight_decay)):
            with tf.variable_scope(scope, 'bev_vgg_lfe', [inputs]) as sc:
                end_points_collection = sc.name + '_end_points'

                # Collect outputs for conv2d, fully_connected and max_pool2d.
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    outputs_collections=end_points_collection):

                    padded = tf.pad(inputs, [[0, 0], [4, 0], [0, 0], [0, 0]])

                    # Encoder
                    conv1 = self.dilated_conv_block(padded, 
                                               vgg_config.vgg_conv1[1],
                                               vgg_config.vgg_conv1[0],
                                               vgg_config.vgg_conv1[2], 
                                               'conv1')

                    conv2 = self.dilated_conv_block(conv1, 
                                               vgg_config.vgg_conv2[1],
                                               vgg_config.vgg_conv2[0],
                                               vgg_config.vgg_conv2[2], 
                                               'conv2')

                    conv3 = self.dilated_conv_block(conv2, 
                                               vgg_config.vgg_conv3[1],
                                               vgg_config.vgg_conv3[0],
                                               vgg_config.vgg_conv3[2], 
                                               'conv3')
                    """
                    slim.repeat(padded,
                                        vgg_config.vgg_conv1[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv1[1],
                                        [3, 3],
                                        rate=vgg_config.vgg_conv1[2],
                                        scope='conv1')
                    

                    conv2 = slim.repeat(conv1,
                                        vgg_config.vgg_conv2[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv2[1],
                                        [3, 3],
                                        rate=vgg_config.vgg_conv2[2],
                                        scope='conv2')

                    conv3 = slim.repeat(conv2,
                                        vgg_config.vgg_conv3[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv3[1],
                                        [3, 3],
                                        rate=vgg_config.vgg_conv3[2],
                                        scope='conv3')
                    """

                    if len(vgg_config.vgg_conv4) == 0:
                        feature_maps_out = conv3
                    else:
                        conv4 = self.dilated_conv_block(conv3, 
                                               vgg_config.vgg_conv4[1],
                                               vgg_config.vgg_conv4[0],
                                               vgg_config.vgg_conv4[2], 
                                               'conv4')
                        feature_maps_out = conv4

                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

                return feature_maps_out, end_points
