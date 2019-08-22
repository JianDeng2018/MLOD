"""Detection model trainer and evaluator.

This runs the detection model trainer and evaluator sequentially.
"""

import argparse
import os
import tensorflow as tf

import mlod
import mlod.builders.config_builder_util as config_builder
from mlod.builders.dataset_builder import DatasetBuilder
from mlod.core.models.mlod_model import MlodModel
from mlod.core.models.rpn_model import RpnModel
from mlod.core import trainer
from mlod.core import evaluator

tf.logging.set_verbosity(tf.logging.ERROR)


def train_and_eval(model_config, train_config, eval_config, dataset_config):

    # Dataset Configuration
    dataset_config_train = DatasetBuilder.copy_config(dataset_config)
    dataset_config_eval = DatasetBuilder.copy_config(dataset_config)

    dataset_train = DatasetBuilder.build_kitti_dataset(dataset_config_train,
                                                       use_defaults=False)
    dataset_eval = DatasetBuilder.build_kitti_dataset(dataset_config_eval,
                                                      use_defaults=False)

    model_name = model_config.model_name
    train_val_test = 'train'
    eval_mode = eval_config.eval_mode
    if eval_mode == 'train':
        raise ValueError('Evaluation mode can only be set to `val` or `test`.')

    # keep a copy as this will be overwritten inside
    # the training loop below
    max_train_iter = train_config.max_iterations
    checkpoint_interval = train_config.checkpoint_interval
    eval_interval = eval_config.eval_interval

    if eval_interval < checkpoint_interval or \
            (eval_interval % checkpoint_interval) != 0:
        raise ValueError(
            'Checkpoint interval (given {}) must be greater than and'
            'divisible by the evaluation interval (given {}).'.format(
                eval_interval, checkpoint_interval))

    # Use the evaluation losses file to continue from the latest
    # checkpoint
    already_evaluated_ckpts = evaluator.get_evaluated_ckpts(model_config,
                                                            model_name)

    if len(already_evaluated_ckpts) != 0:
        current_train_iter = already_evaluated_ckpts[-1]
    else:
        current_train_iter = eval_interval

    # while training is not finished
    while current_train_iter <= max_train_iter:
        # Train
        with tf.Graph().as_default():
            if model_name == 'mlod_model':
                model = MlodModel(model_config,
                                  train_val_test=train_val_test,
                                  dataset=dataset_train)
            elif model_name == 'rpn_model':
                model = RpnModel(model_config,
                                 train_val_test=train_val_test,
                                 dataset=dataset_train)
            else:
                raise ValueError('Invalid model name {}'.format(model_name))

            # overwrite the training epochs
            train_config.max_iterations = current_train_iter
            print('\n*************** Training ****************\n')
            trainer.train(model, train_config)
        current_train_iter += eval_interval

        # Evaluate
        with tf.Graph().as_default():
            if model_name == 'mlod_model':
                model = MlodModel(model_config, train_val_test=eval_mode,
                                  dataset=dataset_eval)
            elif model_name == 'rpn_model':
                model = RpnModel(model_config, train_val_test=eval_mode,
                                 dataset=dataset_eval)
            else:
                raise ValueError('Invalid model name {}'.format(model_name))

            print('\n*************** Evaluating *****************\n')
            evaluator.run_latest_checkpoints(model, dataset_config_eval)
    print('\n************ Finished training and evaluating *************\n')


def main(_):

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    parser = argparse.ArgumentParser()

    default_train_config_path = mlod.root_dir() + \
        '/configs/mlod_exp_example.config'
    parser.add_argument('--pipeline_config',
                        type=str,
                        dest='pipeline_config_path',
                        default=default_train_config_path,
                        help='Path to the train config')

    args = parser.parse_args()

    model_config, train_config, eval_config, dataset_config = \
        config_builder.get_configs_from_pipeline_file(
            args.pipeline_config_path,
            is_training=True)
    train_and_eval(model_config, train_config, eval_config, dataset_config)


if __name__ == '__main__':
    tf.app.run()
