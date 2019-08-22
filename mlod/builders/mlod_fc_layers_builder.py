import tensorflow as tf

from mlod.core.mlod_fc_layers import basic_fc_layers
from mlod.core.mlod_fc_layers import combined_fc_layers
from mlod.core.mlod_fc_layers import fusion_fc_layers


KEY_CLS_LOGITS = 'classification_logits'
KEY_SUB_CLS_LOGITS_LIST = 'sub_classification_logits_list'
KEY_SUB_REG_OFFSETS_LIST = 'sub_regression_offset_list'
KEY_OFFSETS = 'offsets'
KEY_ANGLE_VECTORS = 'angle_vectors'
KEY_ENDPOINTS = 'end_points'
KEY_SELECTED_IMG_IDX = 'selected_image_indices'


def build(layers_config,
          input_rois, cls_input_weights,
          reg_input_weights, num_final_classes, box_rep,
          top_anchors, ground_plane,
          is_training,
          variables_name,
          multi_check,
          net_out,
          img_idx,
          selected_img_idx_in=None):
    """Builds second stage fully connected layers

    Args:
        layers_config: Configuration object
        input_rois: List of input ROI feature maps
        input_weights: List of weights for each input e.g. [1.0, 1.0]
        num_final_classes: Number of output classes, including 'Background'
        box_rep: Box representation (e.g. 'box_3d', 'box_8c', etc.)
        top_anchors: Top proposal anchors, to include location information
        is_training (bool): Whether the network is training or evaluating
        variables_name: name of variable scope
        multi_check (bool): using extra loss function to avoid over fitting
        net_out (list of string): list of output layers

    Returns:
        fc_output_layers: Output layer dictionary
    """

    # Default all output layers to None
    cls_logits = offsets = angle_vectors = end_points = None

    with tf.variable_scope(variables_name) as sc:
        end_points_collection = sc.name + '_end_points'

        fc_layers_type = layers_config.WhichOneof('fc_layers')

        if fc_layers_type == 'basic_fc_layers':
            fc_layers_config = layers_config.basic_fc_layers

            cls_logits, offsets, angle_vectors, end_points = \
                basic_fc_layers.build(
                    fc_layers_config=fc_layers_config,
                    input_rois=input_rois,
                    input_weights=input_weights,
                    num_final_classes=num_final_classes,
                    box_rep=box_rep,

                    is_training=is_training,
                    end_points_collection=end_points_collection)

        elif fc_layers_type == 'combined_fc_layers':
            fc_layers_config = layers_config.combined_fc_layers

            cls_logits, offsets, angle_vectors, end_points = \
                combined_fc_layers.build(
                    fc_layers_config=fc_layers_config,
                    input_rois=input_rois,
                    input_weights=input_weights,
                    num_final_classes=num_final_classes,
                    box_rep=box_rep,
                    top_anchors=top_anchors,
                    ground_plane=ground_plane,

                    is_training=is_training,
                    end_points_collection=end_points_collection)

        elif fc_layers_type == 'fusion_fc_layers':
            fc_layers_config = layers_config.fusion_fc_layers
            cls_logits, offsets, angle_vectors, end_points, sub_cls_logits_list, sub_reg_offset_list, selected_img_idx_out = \
                    fusion_fc_layers.build(
                        fc_layers_config=fc_layers_config,
                        input_rois=input_rois,
                        cls_input_weights=cls_input_weights,
                        reg_input_weights=reg_input_weights,
                        num_final_classes=num_final_classes,
                        box_rep=box_rep,
                        is_training=is_training,
                        end_points_collection=end_points_collection,
                        multi_check=multi_check,
                        net_out=net_out,
                        img_idx=img_idx,
                        selected_img_idx=selected_img_idx_in)
        else:
            raise ValueError('Invalid fc layers config')

    # # Histogram summaries
    # with tf.variable_scope('histograms_mlod'):
    #     for fc_layer in end_points:
    #         tf.summary.histogram(fc_layer, end_points[fc_layer])

    fc_output_layers = dict()
    fc_output_layers[KEY_CLS_LOGITS] = cls_logits
    fc_output_layers[KEY_OFFSETS] = offsets

    fc_output_layers[KEY_SUB_CLS_LOGITS_LIST] = sub_cls_logits_list
    fc_output_layers[KEY_SUB_REG_OFFSETS_LIST] = sub_reg_offset_list
    fc_output_layers[KEY_SELECTED_IMG_IDX] = selected_img_idx_out
    fc_output_layers[KEY_ANGLE_VECTORS] = angle_vectors
    fc_output_layers[KEY_ENDPOINTS] = end_points
    

    return fc_output_layers
