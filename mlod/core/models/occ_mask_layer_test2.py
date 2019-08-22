import argparse
import os
import cv2

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

import mlod
import mlod.builders.config_builder_util as config_builder
from mlod.builders.dataset_builder import DatasetBuilder
from wavedata.tools.obj_detection import obj_utils
from mlod.core import constants

from occlusion_mask_layer import OccMaskLayer

parser = argparse.ArgumentParser()

default_pipeline_config_path = mlod.root_dir() + \
    '/configs/mlod_fpn_people.config'

parser.add_argument('--pipeline_config',
                    type=str,
                    dest='pipeline_config_path',
                    default=default_pipeline_config_path,
                    help='Path to the pipeline config')

parser.add_argument('--device',
                    type=str,
                    dest='device',
                    default='0',
                    help='CUDA device id')

checkpoint_to_restore = ''

args = parser.parse_args()

# Parse pipeline config
model_config, _, eval_config, dataset_config = \
    config_builder.get_configs_from_pipeline_file(
        args.pipeline_config_path,
        is_training=False)

# Overwrite data split
dataset_config.data_split = 'trainval'

dataset_config.data_split_dir = 'training'
dataset_config.has_labels = True

# Set CUDA device id
os.environ['CUDA_VISIBLE_DEVICES'] = args.device

# Convert to object to overwrite repeated fields
dataset_config = config_builder.proto_to_obj(dataset_config)

# Remove augmentation during evaluation
dataset_config.aug_list = ['flipping']

# Build the dataset object
dataset = DatasetBuilder.build_kitti_dataset(dataset_config,
                                             use_defaults=False)

# Setup the model
model_name = model_config.model_name

# Convert to object to overwrite repeated fields
model_config = config_builder.proto_to_obj(model_config)

# Switch path drop off during evaluation
model_config.path_drop_probabilities = [1.0, 1.0]

index = 76
samples = dataset.load_samples([index])
depth_map= samples[0].get(constants.KEY_DPT_INPUT)
print(depth_map.shape)
image_input = samples[0].get(constants.KEY_IMAGE_INPUT)
h = image_input.shape[0]
w = image_input.shape[1]
obj_position = [236.84, 169.82, 288.56, 339.57]

boxes_norm = tf.Variable([[170/h, 236/w, 340/h, 289/w]])

depth_ph = tf.placeholder(tf.float32, (1, None,None), 'depth_input')

print(h,w)

n_split = 8
img_size = (6,6)
ref_depth_min = tf.Variable([5.3])
ref_depth_max = tf.Variable([7.3])

depth_input = tf.expand_dims(depth_ph,-1)
box_indices = tf.zeros([1],dtype=tf.int32)
occ_mak_layer = OccMaskLayer()
depth_val, occ_mask = occ_mak_layer.build(depth_input,boxes_norm,box_indices,ref_depth_min,ref_depth_max,n_split,[8,8],img_size,0.5)

init_op = tf.global_variables_initializer()
#sess= tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess = tf.Session()
sess.run(init_op)
#print(sess.run(occ_mask,feed_dict={img_ph:[img0],depth_ph:depth_map}))
depth = sess.run(depth_val,feed_dict={depth_ph:depth_map})
print(depth)