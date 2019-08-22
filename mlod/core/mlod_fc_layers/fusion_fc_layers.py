from tensorflow.contrib import slim
import tensorflow as tf
from mlod.core.mlod_fc_layers import mlod_fc_layer_utils
import numpy as np

def build(fc_layers_config,
          input_rois, cls_input_weights,
          reg_input_weights, num_final_classes, box_rep,
          is_training,
          end_points_collection,
          multi_check,
          net_out,
          img_idx,
          selected_img_idx):
    """Builds fusion layers

    Args:
        fc_layers_config: Fully connected layers config object
        input_rois: List of input roi feature maps
        input_weights: List of weights for each input e.g. [1.0, 1.0]
        num_final_classes: Final number of output classes, including
            'Background'
        box_rep: Box representation (e.g. 'box_3d', 'box_8c', 'box_4c')
        is_training: Whether the network is training or evaluating
        end_points_collection: End points collection to add entries to

    Returns:
        cls_logits: Output classification logits
        offsets: Output offsets
        angle_vectors: Output angle vectors (or None)
        end_points: End points dict
    """

    # Parse configs
    fusion_type = fc_layers_config.fusion_type
    fusion_method = fc_layers_config.fusion_method

    num_layers = fc_layers_config.num_layers
    layer_sizes = fc_layers_config.layer_sizes
    l2_weight_decay = fc_layers_config.l2_weight_decay
    keep_prob = fc_layers_config.keep_prob

    # Validate values
    #if not len(cls_input_weights) == len(input_rois) == len(reg_input_weights):
    #    raise ValueError('Length of input_weights does not match length of '
    #                     'input_rois')
    if not len(layer_sizes) == num_layers:
        raise ValueError('Length of layer_sizes does not match num_layers')

    sub_cls_logits_list = []

    if fusion_type == 'early':
        cls_logits, offsets, angle_vectors, sub_cls_logits_list, sub_reg_offset_list = \
            _early_fusion_fc_layers(num_layers=num_layers,
                                    layer_sizes=layer_sizes,
                                    input_rois=input_rois,
                                    input_weights=cls_input_weights,
                                    fusion_method=fusion_method,
                                    l2_weight_decay=l2_weight_decay,
                                    keep_prob=keep_prob,
                                    num_final_classes=num_final_classes,
                                    box_rep=box_rep,
                                    is_training=is_training,
                                    net_out=net_out,
                                    multi_check=multi_check,
                                    selected_img_idx=selected_img_idx,
                                    img_idx=img_idx)

    elif fusion_type == 'late':
      cls_logits, offsets, angle_vectors, sub_cls_logits_list, sub_reg_offset_list = \
          _late_fusion_fc_layers(num_layers=num_layers,
                                 layer_sizes=layer_sizes,
                                 input_rois=input_rois,
                                 input_weights=cls_input_weights,
                                 fusion_method=fusion_method,
                                 l2_weight_decay=l2_weight_decay,
                                 keep_prob=keep_prob,
                                 num_final_classes=num_final_classes,
                                 box_rep=box_rep,
                                 is_training=is_training,
                                 net_out=net_out,
                                 multi_check=multi_check,
                                 selected_img_idx=selected_img_idx,
                                 img_idx=img_idx)

    elif fusion_type == 'deep':
        cls_logits, offsets, angle_vectors = \
            _deep_fusion_fc_layers(num_layers=num_layers,
                                   layer_sizes=layer_sizes,
                                   input_rois=input_rois,
                                   input_weights=cls_input_weights,
                                   fusion_method=fusion_method,
                                   l2_weight_decay=l2_weight_decay,
                                   keep_prob=keep_prob,
                                   num_final_classes=num_final_classes,
                                   box_rep=box_rep,
                                   is_training=is_training)
    else:
        raise ValueError('Invalid fusion type {}'.format(fusion_type))

    # Convert to end point dict
    end_points = slim.utils.convert_collection_to_dict(end_points_collection)

    return cls_logits, offsets, angle_vectors, end_points, sub_cls_logits_list, sub_reg_offset_list, selected_img_idx

def build_output_layers(tensor_in,
                        num_final_classes,
                        box_rep,
                        net_out,
                        img_idx,
                        branch_idx=None):
    """Builds flattened output layers

    Args:
        tensor_in: Input tensor
        num_final_classes: Final number of output classes, including
            'Background'
        box_rep: Box representation (e.g. 'box_3d', 'box_8c', 'box_4c')

    Returns:
        Output layers
    """

    # get branch name
    if branch_idx is None:
      #fusion branch
      branch_name = ''
    else:
      branch_name = '_'+str(branch_idx)

    # Classification
    if 'cls' in net_out:
      cls_logits = build_output(tensor_in,num_final_classes, 'cls_out'+branch_name )
    else:
      cls_logits = None

    # Offsets
    if branch_idx in img_idx:
      # img offset
      off_out_size = 4
    else:
      off_out_size = mlod_fc_layer_utils.OFFSETS_OUTPUT_SIZE[box_rep]
    
    if off_out_size > 0 and 'offset' in net_out:
      off_out = build_output(tensor_in, off_out_size*num_final_classes, 'off_out'+branch_name)
    else:
      off_out = None

    # Angle Unit Vectors
    ang_out_size = mlod_fc_layer_utils.ANG_VECS_OUTPUT_SIZE[box_rep]
    # skip for sub_branches
    if ang_out_size > 0 and 'ang' in net_out and branch_idx is None:
        ang_out = build_output(tensor_in,ang_out_size, 'ang_out'+branch_name)
    else:
        ang_out = None

    return cls_logits, off_out, ang_out

def build_output(tensor_in,num_output, scope_name):
  
  output = slim.fully_connected(tensor_in, num_output, 
                              activation_fn=None,
                              scope=scope_name)
  return output

def _early_fusion_fc_layers(num_layers, layer_sizes,
                            input_rois, input_weights, fusion_method,
                            l2_weight_decay, keep_prob,
                            num_final_classes, box_rep,
                            is_training,net_out, multi_check,img_idx,selected_img_idx):

    if not num_layers == len(layer_sizes):
        raise ValueError('num_layers does not match length of layer_sizes')

    if l2_weight_decay > 0:
        weights_regularizer = slim.l2_regularizer(l2_weight_decay)
    else:
        weights_regularizer = None

    branch_img_outputs = []
    branch_bev_outputs = []
    branch_outputs = []
    
    sub_cls_logits_list = []
    sub_reg_offset_list = []
    sub_img_cls_logits_list = []

    num_branches = len(input_rois)

    with slim.arg_scope(
          [slim.fully_connected],
          weights_regularizer=weights_regularizer):
      for branch_idx in range(num_branches):

          # Branch feature ROIs
          with tf.variable_scope('br{}'.format(branch_idx)):
            branch_rois = input_rois[branch_idx]

            rois_flatten = slim.flatten(branch_rois,scope='flatten')

            if multi_check:
              fc_drop = slim.dropout(
                      rois_flatten,
                      keep_prob=keep_prob,
                      is_training=is_training,
                      scope='fc_br{}_drop'.format(branch_idx))

              fc_layer = slim.fully_connected(fc_drop, layer_sizes[0],
                                            scope='fc_br{}'.format(branch_idx))

              sub_cls_logits, sub_off_out, _ = build_output_layers(fc_layer,
                                        num_final_classes,
                                        box_rep,
                                        net_out,
                                        img_idx,
                                        branch_idx)

              sub_cls_logits_list.append(sub_cls_logits)
              sub_reg_offset_list.append(sub_off_out)
              if branch_idx in img_idx:
                sub_img_cls_logits_list.append(sub_cls_logits)

            # save the channel separaredly 
            if branch_idx in img_idx:
              branch_img_outputs.append(fc_layer)
            else:
              branch_bev_outputs.append(fc_layer)

    # use multi-check and more than one channel
    if multi_check and len(sub_img_cls_logits_list) > 1:
      sub_cls_logits_tensor = tf.stack(sub_img_cls_logits_list,axis=0)
      selected_img_idx, selected_img_scale = mlod_fc_layer_utils.scale_selection(selected_img_idx, sub_cls_logits_tensor, branch_img_outputs)
      selected_img_output = selected_img_scale

      max_img_logits = tf.gather_nd(sub_cls_logits_tensor,selected_img_idx)
      sub_cls_logits_list.append(max_img_logits)
    else:
      selected_img_output = branch_img_outputs[0]

    branch_outputs = [branch_bev_outputs[0], selected_img_output]

    # Feature fusion
    fused_features = mlod_fc_layer_utils.feature_fusion(fusion_method,
                                                        branch_outputs,
                                                        input_weights)

    #res_feature = res_layer(fused_features,is_training, layer_sizes)

    res_feature = fc_drop_layer(fused_features, is_training, layer_sizes, weights_regularizer,keep_prob)

    cls_logits, off_out, ang_out = build_output_layers(res_feature,
                                        num_final_classes,
                                        box_rep,
                                        net_out,
                                        img_idx)

    return cls_logits, off_out, ang_out, sub_cls_logits_list,sub_reg_offset_list


def _late_fusion_fc_layers(num_layers, layer_sizes,
                           input_rois, input_weights, fusion_method,
                           l2_weight_decay, keep_prob,
                           num_final_classes, box_rep,
                           is_training,net_out, multi_check, img_idex, selected_img_idx):

    if l2_weight_decay > 0:
        weights_regularizer = slim.l2_regularizer(l2_weight_decay)
    else:
        weights_regularizer = None

    # Build fc layers, one branch per input
    num_branches = len(input_rois)
    branch_img_outputs = []
    branch_bev_outputs = []

    sub_cls_logits_list = []
    sub_reg_offset_list = []
    sub_img_cls_logits_list = []

    with slim.arg_scope(
            [slim.fully_connected],
            weights_regularizer=weights_regularizer):
        for branch_idx in range(num_branches):

            # Branch feature ROIs
            with tf.variable_scope('br{}'.format(branch_idx)):
              branch_rois = input_rois[branch_idx]
              fc_drop = slim.flatten(branch_rois,
                                     scope='flatten')

              for layer_idx in range(num_layers):
                  fc_name_idx = 6 + layer_idx

                  # Use conv2d instead of fully_connected layers.
                  fc_layer = slim.fully_connected(
                      fc_drop, layer_sizes[layer_idx],
                      scope='fc{}'.format(branch_idx, fc_name_idx))

                  fc_drop = slim.dropout(
                      fc_layer,
                      keep_prob=keep_prob,
                      is_training=is_training,
                      scope='fc{}_drop'.format(branch_idx, fc_name_idx))

                  if multi_check:
                    sub_cls_logits, sub_off_out, _ = build_output_layers(fused_features,
                                              num_final_classes,
                                              box_rep,
                                              net_out,
                                              img_idx,
                                              branch_idx)

                    sub_cls_logits_list.append(sub_cls_logits)
                    sub_reg_offset_list.append(sub_off_out)
                    if branch_idx in img_idx:
                      sub_img_cls_logits_list.append(sub_cls_logits)

                  # save the channel separaredly 
                  if branch_idx in img_idx:
                    branch_img_outputs.append(fc_layer)
                  else:
                    branch_bev_outputs.append(fc_layer)

        # use multi-check and more than one channel
        if multi_check and len(sub_img_cls_logits_list) > 1:
          sub_cls_logits_tensor = tf.stack(sub_img_cls_logits_list,axis=0)
          selected_img_idx, selected_img_scale = mlod_fc_layer_utils.scale_selection(selected_img_idx, sub_cls_logits_tensor, branch_img_outputs)
          selected_img_output = selected_img_scale

          max_img_logits = tf.gather_nd(sub_cls_logits_tensor,selected_img_idx)
          sub_cls_logits_list.append(max_img_logits)
        else:
          selected_img_output = branch_img_outputs[0]

        branch_outputs = [branch_bev_outputs[0], selected_img_output]

        # Feature fusion
        fused_features = mlod_fc_layer_utils.feature_fusion(fusion_method,
                                                            branch_outputs,
                                                            input_weights)

        # Ouput layers
        cls_logits, off_out, ang_out = build_output_layers(fused_features,
                                            num_final_classes,
                                            box_rep,
                                            net_out,
                                            img_idx)
    return cls_logits, off_out, ang_out, sub_cls_logits_list,sub_reg_offset_list

def _deep_fusion_fc_layers(num_layers, layer_sizes,
                           input_rois, input_weights, fusion_method,
                           l2_weight_decay, keep_prob,
                           num_final_classes, box_rep,
                           is_training):

    if l2_weight_decay > 0:
        weights_regularizer = slim.l2_regularizer(l2_weight_decay)
    else:
        weights_regularizer = None

    # Apply fusion
    fusion_layer = mlod_fc_layer_utils.feature_fusion(fusion_method,
                                                      input_rois,
                                                      input_weights)
    fusion_layer = slim.flatten(fusion_layer, scope='flatten')

    with slim.arg_scope(
            [slim.fully_connected],
            weights_regularizer=weights_regularizer):
        # Build layers
        for layer_idx in range(num_layers):
            fc_name_idx = 6 + layer_idx

            all_branches = []
            for branch_idx in range(len(input_rois)):
                fc_layer = slim.fully_connected(
                    fusion_layer, layer_sizes[layer_idx],
                    scope='br{}_fc{}'.format(branch_idx, fc_name_idx))
                fc_drop = slim.dropout(
                    fc_layer,
                    keep_prob=keep_prob,
                    is_training=is_training,
                    scope='br{}_fc{}_drop'.format(branch_idx, fc_name_idx))

                all_branches.append(fc_drop)

            # Apply fusion
            fusion_layer = mlod_fc_layer_utils.feature_fusion(fusion_method,
                                                              all_branches,
                                                              input_weights)

        # Ouput layers
        output_layers = build_output_layers(fusion_layer,
                                            num_final_classes,
                                            box_rep)
    return output_layers

def res_layer(input_layer,is_training, layer_sizes):
    start_layer = slim.fully_connected(input_layer,
                                       layer_sizes[0], 
                                       normalizer_fn=slim.batch_norm,
                                       normalizer_params={
                                       'is_training': is_training})
    layer = start_layer
    for i in range(len(layer_sizes)):
      bn_features = slim.batch_norm(layer, is_training=is_training)
      relu_feature = tf.nn.relu(bn_features)
      layer = slim.fully_connected(relu_feature,layer_sizes[i], activation_fn=None)

    return start_layer + layer

def fc_drop_layer(input_layer,is_training,layer_sizes, weights_regularizer,keep_prob):
  fc_drop = input_layer
  with slim.arg_scope(
            [slim.fully_connected],
            weights_regularizer=weights_regularizer):

        for layer_idx in range(len(layer_sizes)):
            fc_name_idx = 6 + layer_idx

            # Use conv2d instead of fully_connected layers.
            fc_layer = slim.fully_connected(fc_drop, layer_sizes[layer_idx],
                                            scope='fc{}'.format(fc_name_idx))

            fc_drop = slim.dropout(
                fc_layer,
                keep_prob=keep_prob,
                is_training=is_training,
                scope='fc{}_drop'.format(fc_name_idx))

            fc_name_idx += 1
  return fc_drop
