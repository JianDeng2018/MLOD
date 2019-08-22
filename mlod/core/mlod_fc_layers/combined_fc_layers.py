import tensorflow as tf
from tensorflow.contrib import slim

from mlod.core.mlod_fc_layers import mlod_fc_layer_utils


def build(fc_layers_config,
          input_rois, input_weights,
          num_final_classes, box_rep,
          top_anchors, ground_plane,

          is_training,
          end_points_collection):
    """Builds experimental combined fc layers
    """

    # Parse config
    fusion_method = fc_layers_config.fusion_method
    l2_weight_decay = fc_layers_config.l2_weight_decay

    fusion_out = mlod_fc_layer_utils.feature_fusion(fusion_method,
                                                    input_rois,
                                                    input_weights)

    with tf.variable_scope('flatten'):
        flattened = slim.flatten(fusion_out)

    with tf.variable_scope('append_features'):
        # Append proposal positions as features
        # Normalize anchor positions along
        # x: [-40, 40], z: [0, 70]
        # Z positions are reversed to match depth map input
        top_anchors_x = tf.reshape(top_anchors[:, 0], (-1, 1)) / 40.0
        top_anchors_z = 1.0 - (tf.reshape(top_anchors[:, 2], (-1, 1)) / 70.0)

        # Concatenate features to a flattened feature vector
        fc_tensor_in = tf.concat(
            [flattened, top_anchors_x, top_anchors_z], axis=1)

    # L2 Regularizer
    if l2_weight_decay > 0:
        weights_regularizer = slim.l2_regularizer(l2_weight_decay)
    else:
        weights_regularizer = None

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=weights_regularizer,
                        outputs_collections=end_points_collection):
        tensor_in = fc_tensor_in

        with tf.variable_scope('fc_layers'):
            fc6 = slim.fully_connected(tensor_in, fc_layers_config.fc6,
                                       scope='fc6')
            fc6_drop = slim.dropout(fc6,
                                    fc_layers_config.keep_prob,
                                    is_training=is_training,
                                    scope='fc6_drop')

            fc7 = slim.fully_connected(fc6_drop, fc_layers_config.fc7,
                                       scope='fc7')
            fc7_drop = slim.dropout(fc7,
                                    fc_layers_config.keep_prob,
                                    is_training=is_training,
                                    scope='fc7_drop')

            fc8 = slim.fully_connected(fc7_drop, fc_layers_config.fc8,
                                       scope='fc8')
            fc8_drop = slim.dropout(fc8,
                                    fc_layers_config.keep_prob,
                                    is_training=is_training,
                                    scope='fc8_drop')

        with tf.variable_scope('classifications'):
            cls_logits = slim.fully_connected(
                fc8_drop,
                num_final_classes,
                activation_fn=None,
                scope='cls_out')

        offsets = None
        offsets_output_size = mlod_fc_layer_utils.OFFSETS_OUTPUT_SIZE[box_rep]
        if offsets_output_size > 0:
            with tf.variable_scope('offsets'):
                # Offsets Prediction
                offsets = slim.fully_connected(
                    fc8_drop,
                    offsets_output_size,
                    activation_fn=None,
                    scope='off_out')

        angle_vectors = None
        angle_vectors_output_size = \
            mlod_fc_layer_utils.ANG_VECS_OUTPUT_SIZE[box_rep]
        if angle_vectors_output_size > 0:
            with tf.variable_scope('angle_vectors'):
                # Orientation Angle Prediction (unit vector)
                angle_vectors = slim.fully_connected(
                    fc8_drop,
                    angle_vectors_output_size,
                    activation_fn=None,
                    scope='ang_out')

    end_points = slim.utils.convert_collection_to_dict(end_points_collection)
    return cls_logits, offsets, angle_vectors, end_points
