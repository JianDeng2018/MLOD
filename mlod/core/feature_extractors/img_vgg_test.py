"""VGG Network Test."""

import os
import numpy as np
import tensorflow as tf
import unittest

import mlod
from mlod.datasets.imagenet import dataset_utils
from mlod.core.feature_extractors import img_vgg

slim = tf.contrib.slim


@unittest.skip("skip loading img_vgg weights")
class ImgVGGTest(tf.test.TestCase):

    def setUp(self):
        tf.test.TestCase.setUp(self)

        img_feature_extractor = img_vgg.ImgVgg16()

        # dummy imageplaceholder
        img_input_placeholder = tf.placeholder(tf.float32, [480, 1590, 3])
        img_input_batches = tf.expand_dims(img_input_placeholder, axis=0)

        img_preprocessed = img_feature_extractor.preprocess_input(
                                                img_input_batches,
                                                [224, 224])

        _, img_end_points = img_feature_extractor.build(
                                             img_preprocessed)

        # download the vgg weights
        url = "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"
        self.vgg_checkpoint_dir = mlod.root_dir() + '/checkpoints/vgg_16'

        if not tf.gfile.Exists(self.vgg_checkpoint_dir):
            tf.gfile.MakeDirs(self.vgg_checkpoint_dir)

        # Download vgg16 if it hasn't been downloaded yet
        if not os.path.isfile(
                self.vgg_checkpoint_dir +
                "/vgg_16_2016_08_28.tar.gz"):
            dataset_utils.download_and_uncompress_tarball(
                url, self.vgg_checkpoint_dir)
        else:
            print("Already downloaded")

    def test_vgg_weights(self):

        first_load_weights = []
        snd_load_weights = []
        init_op = tf.global_variables_initializer()
        with self.test_session() as sess:
            sess.run(init_op)
            # Initialize vgg16 weights
            init_fn = slim.assign_from_checkpoint_fn(
                      os.path.join(self.vgg_checkpoint_dir,
                                   'vgg_16.ckpt'),
                                   slim.get_model_variables('vgg_16'),
                                   ignore_missing_vars=True)
            init_fn(sess)
            vgg_vars = slim.get_model_variables('vgg_16')
            first_load_weights = sess.run(vgg_vars)

        with self.test_session() as sess:
            sess.run(init_op)
            # Initialize vgg16 weights again
            init_fn = slim.assign_from_checkpoint_fn(
                      os.path.join(self.vgg_checkpoint_dir,
                                   'vgg_16.ckpt'),
                                   slim.get_model_variables('vgg_16'),
                                   ignore_missing_vars=True)
            init_fn(sess)
            vgg_vars = slim.get_model_variables('vgg_16')
            snd_load_weights = sess.run(vgg_vars)

        for i in range(len(first_load_weights)):
            np.testing.assert_array_equal(first_load_weights[i],
                                          snd_load_weights[i])


if __name__ == '__main__':
    tf.test.main()
