import os

import numpy as np

try:
    import urllib2
except ImportError:
    import urllib.request as urllib

import mlod
from mlod.datasets.imagenet import dataset_utils
from mlod.core.feature_extractors import vgg_preprocessing, vgg

from tensorflow.contrib import slim
import tensorflow as tf


def vis_feature_maps(end_points, layer_name):

    feature_maps = end_points.get(layer_name)

    with tf.name_scope(layer_name):
        batch, map_width, map_height, num_maps = np.array(
            feature_maps.shape).astype(np.int32)

        # Take first map only
        output = tf.slice(feature_maps, (0, 0, 0, 0), (1, -1, -1, -1))
        output = tf.reshape(output, (map_height, map_width, num_maps))

        # Add padding around each map
        map_width += 5
        map_height += 5
        output = tf.image.resize_image_with_crop_or_pad(
            output, map_height, map_width)

        # Find good image size for display
        map_sizes = [64, 128, 256, 512]
        image_sizes = [(8, 8), (16, 8), (16, 16), (32, 16)]
        size_idx = map_sizes.index(num_maps)
        desired_image_size = image_sizes[size_idx]
        image_width = desired_image_size[0]
        image_height = desired_image_size[1]

        # Arrange maps into a grid
        output = tf.reshape(output, (map_height, map_width, image_height,
                                     image_width))
        output = tf.transpose(output, (2, 0, 3, 1))
        output = tf.reshape(output, (1, image_height * map_height,
                                     image_width * map_width, 1))

        layer_name = layer_name.split('/')[-1]
        tf.summary.image(layer_name, output, max_outputs=10)


def inference(sess, image):

    # Download VGG16
    url = "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"
    # checkpoints_dir = '/tmp/mlod/checkpoints'
    checkpoints_dir = mlod.root_dir() + '/checkpoints/vgg_16'

    if not tf.gfile.Exists(checkpoints_dir):
        tf.gfile.MakeDirs(checkpoints_dir)

    # Download vgg16 if it hasn't been downloaded yet
    if not os.path.isfile(checkpoints_dir + "/vgg_16_2016_08_28.tar.gz"):
        dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)
    else:
        print("Already downloaded")

    # Set the image size to [224, 224]
    image_size = vgg.vgg_16.default_image_size

    # Pre-process input
    with tf.name_scope('input_reshape'):
        processed_image = vgg_preprocessing.preprocess_image(
            image, image_size, image_size, is_training=False)

        image_summary = tf.expand_dims(image, 0)
        processed_images = tf.expand_dims(processed_image, 0)

        tf.summary.image('images', image_summary, max_outputs=5)
        tf.summary.image('processed_images', processed_images, max_outputs=5)

    # Create the model, use the default arg scope to configure the
    # batch norm parameters.
    with slim.arg_scope(vgg.vgg_arg_scope()):
        # 1000 classes instead of 1001.
        logits, end_points = vgg.vgg_16(
            processed_images, num_classes=1000, is_training=False)
    probabilities = tf.nn.softmax(logits)

    # Add some images for feature maps from conv layers
    vis_feature_maps(end_points, 'vgg_16/conv1/conv1_1')
    vis_feature_maps(end_points, 'vgg_16/conv2/conv2_1')
    vis_feature_maps(end_points, 'vgg_16/conv3/conv3_1')
    vis_feature_maps(end_points, 'vgg_16/conv4/conv4_1')
    vis_feature_maps(end_points, 'vgg_16/conv5/conv5_1')
    vis_feature_maps(end_points, 'vgg_16/conv5/conv5_3')

    # Initialize vgg16 weights
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'vgg_16.ckpt'),
        slim.get_model_variables('vgg_16'))
    init_fn(sess)

    return logits, probabilities
