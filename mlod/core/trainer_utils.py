import os
import tensorflow as tf

from mlod.datasets.imagenet import dataset_utils

slim = tf.contrib.slim


def load_checkpoints(checkpoint_dir, saver):

    # Load latest checkpoint if available
    all_checkpoint_states = tf.train.get_checkpoint_state(
        checkpoint_dir)
    if all_checkpoint_states is not None:
        all_checkpoint_paths = \
            all_checkpoint_states.all_model_checkpoint_paths
        # Save the checkpoint list into saver.last_checkpoints
        saver.recover_last_checkpoints(all_checkpoint_paths)
    else:
        print('No checkpoints found')


def get_global_step(sess, global_step_tensor):
    # Read the global step if restored
    global_step = tf.train.global_step(sess,
                                       global_step_tensor)
    return global_step


def create_dir(dir):
    """
    Checks if a directory exists, or else create it

    Args:
        dir: directory to create
    """
    if not os.path.exists(dir):
        os.makedirs(dir)


def load_model_weights(model, sess, checkpoint_dir):
    """Restores the model weights.

    Loads the weights loaded from checkpoint dir onto the
    model. It ignores the missing weights since this is used
    to load the RPN weights onto MLOD.

    Args:
        model: A DetectionModel instance
        sess: A tensorflow session
        checkpoint_dir: Path to the weights to be loaded
    """

    init_fn = slim.assign_from_checkpoint_fn(
        checkpoint_dir,
        slim.get_model_variables(),
        ignore_missing_vars=True)
    init_fn(sess)


def download_vgg(vgg_checkpoint_dir):

    # download the vgg weights
    url = "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"

    if not tf.gfile.Exists(vgg_checkpoint_dir):
        tf.gfile.MakeDirs(vgg_checkpoint_dir)

    # Download vgg16 if it hasn't been downloaded yet
    if not os.path.isfile(vgg_checkpoint_dir + "/vgg_16_2016_08_28.tar.gz"):
        dataset_utils.download_and_uncompress_tarball(url, vgg_checkpoint_dir)
    else:
        print("Already downloaded")


def initialize_vgg(model_config,
                   sess):
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(model_config.vgg_checkpoint_dir,
                     'vgg_16.ckpt'),
        slim.get_model_variables('vgg_16'),
        ignore_missing_vars=True)
    init_fn(sess)


