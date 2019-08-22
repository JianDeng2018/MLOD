import tensorflow as tf

from mlod.core.feature_extractors import img_feature_extractor

slim = tf.contrib.slim


class ImgVggRtn(img_feature_extractor.ImgFeatureExtractor):
    """Modified VGG model definition to extract features from
    RGB image input using pyramid features.
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

    def build(self,
              inputs,
              input_pixel_size,
              is_training,
              scope='vgg_16'):   # benz, img_vgg_pyr
        """ Modified VGG for image feature extraction with pyramid features.
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
            with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:  # benz, img_vgg_pyr
                end_points_collection = sc.name + '_end_points'

                # Collect outputs for conv2d, fully_connected and max_pool2d.
                with slim.arg_scope([slim.conv2d,
                                     slim.fully_connected,
                                     slim.max_pool2d],
                                    outputs_collections=end_points_collection):

                    # Encoder
                    conv1 = slim.repeat(inputs,
                                        vgg_config.vgg_conv1[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv1[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},#benz
                                        scope='conv1')
                    pool1 = slim.max_pool2d(
                        conv1, [2, 2], scope='pool1')

                    conv2 = slim.repeat(pool1,
                                        vgg_config.vgg_conv2[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv2[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv2')
                    pool2 = slim.max_pool2d(
                        conv2, [2, 2], scope='pool2')

                    conv3 = slim.repeat(pool2,
                                        vgg_config.vgg_conv3[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv3[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv3')
                    pool3 = slim.max_pool2d(
                        conv3, [2, 2], scope='pool3')

                    conv4 = slim.repeat(pool3,
                                        vgg_config.vgg_conv4[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv4[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv4')


                    pool4 = slim.max_pool2d(
                        conv4, [2, 2], scope='pool4')

                    conv5 = slim.repeat(pool4,
                                        vgg_config.vgg_conv5[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv5[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv5')

                    pool5 = slim.max_pool2d(
                        conv5, [2, 2], scope='pool5')

                    # Decoder (upsample and fuse features)
                    p5 = slim.conv2d(pool5, 256, [1, 1], scope='conv5_reduce')
                    pool4_shape = tf.shape(pool4)[1:3]
                    up_p5 = tf.image.resize_bilinear(p5, pool4_shape, name='p5_upsample')
                    p5 = slim.conv2d(up_p5, 256, [3, 3], scope='p5')          

                    p4 = slim.conv2d(pool4, 256, [1, 1], scope='conv4_reduce')
                    p4 = p4 + up_p5
                    pool3_shape = tf.shape(pool3)[1:3]
                    up_p4 = tf.image.resize_bilinear(p4, pool3_shape, name='p4_upsample')
                    p4 = slim.conv2d(up_p4, 256, [3, 3], scope='p4')

                    p3 = slim.conv2d(pool3, 256, [1, 1], scope='conv3_reduce')
                    p3 = up_p4 + p3
                    p3 = slim.conv2d(p3, 256, [3, 3], scope='p3')

                    p6 = slim.conv2d(pool5, 256, [3, 3], stride=2, activation_fn=None, scope='p6')

                    p7 = tf.nn.relu(p6)
                    p7 = slim.conv2d(p7,256,[3,3], stride=2,scope='p7')

                feature_maps_out = p7 #pyramid_fusion1

                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

                return feature_maps_out, end_points

