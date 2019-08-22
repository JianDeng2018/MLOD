"""Detection model trainer.

This runs the DetectionModel stage-wise trainer.
"""

import argparse
import tensorflow as tf
import os

from copy import deepcopy

import mlod
import mlod.builders.config_builder_util as config_builder
from mlod.builders.dataset_builder import DatasetBuilder
from mlod.core.models.mlod_model import MlodModel
from mlod.core.models.rpn_model import RpnModel
from mlod.core import trainer
from mlod.core import trainer_utils

tf.logging.set_verbosity(tf.logging.ERROR)


def train(rpn_model_config, mlod_model_config,
          rpn_train_config, mlod_train_config,
          dataset_config):

    train_val_test = 'train'
    dataset = DatasetBuilder.build_kitti_dataset(dataset_config,
                                                 use_defaults=False)

    paths_config = rpn_model_config.paths_config
    rpn_checkpoint_dir = paths_config.checkpoint_dir

    with tf.Graph().as_default():
        model = RpnModel(rpn_model_config,
                         train_val_test=train_val_test,
                         dataset=dataset)
        trainer.train(model, rpn_train_config)

        # load the weights back in
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            trainer_utils.load_checkpoints(rpn_checkpoint_dir,
                                           saver)
            checkpoint_to_restore = saver.last_checkpoints[-1]
            trainer_utils.load_model_weights(model,
                                             sess,
                                             checkpoint_to_restore)

    # Merge RPN configs with MLOD - This will overwrite
    # the appropriate configs set for MLOD while keeping
    # the common configs the same.
    rpn_model_config.MergeFrom(mlod_model_config)
    rpn_train_config.MergeFrom(mlod_train_config)
    mlod_model_merged = deepcopy(rpn_model_config)
    mlod_train_merged = deepcopy(rpn_train_config)

    with tf.Graph().as_default():
        model = MlodModel(mlod_model_merged,
                          train_val_test=train_val_test,
                          dataset=dataset)
        trainer.train(model, mlod_train_merged,
                      stagewise_training=True,
                      init_checkpoint_dir=rpn_checkpoint_dir)


def main(_):
    parser = argparse.ArgumentParser()

    default_rpn_config_path = mlod.root_dir() + \
        '/configs/stagewise_rpn_example.config'
    parser.add_argument('--pipeline_rpn_config',
                        type=str,
                        dest='rpn_config_path',
                        default=default_rpn_config_path,
                        help='Path to the rpn config')

    default_mlod_config_path = mlod.root_dir() + \
        '/configs/stagewise_mlod_example.config'
    parser.add_argument('--mlod_config_path',
                        type=str,
                        dest='mlod_config_path',
                        default=default_mlod_config_path,
                        help='Path to the mlod config')

    args = parser.parse_args()

    rpn_model_config, rpn_train_config, _,  dataset_config = \
        config_builder.get_configs_from_pipeline_file(
            args.rpn_config_path,
            is_training=True,
            stagewise_training=True)

    mlod_model_config, mlod_train_config, _,  _ = \
        config_builder.get_configs_from_pipeline_file(
            args.mlod_config_path,
            is_training=True,
            stagewise_training=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    train(rpn_model_config, mlod_model_config,
          rpn_train_config, mlod_train_config,
          dataset_config)


if __name__ == '__main__':
    tf.app.run()
