import tensorflow as tf

from mlod.core import losses
from mlod.core import orientation_encoder

KEY_CLASSIFICATION_LOSS = 'classification_loss'
KEY_REGRESSION_LOSS = 'regression_loss'
KEY_MLOD_LOSS = 'mlod_loss'
KEY_SUB_CLASSIFICATION_LOSS = 'sub_classification_loss'
KEY_OFFSET_LOSS_NORM = 'offset_loss_norm'
KEY_ANG_LOSS_NORM = 'ang_loss_norm'



def build(model, prediction_dict):
    """Builds the loss for a variety of box representations

    Args:
        model: network model
        prediction_dict: prediction dictionary

    Returns:
        losses_output: loss dictionary
    """

    mlod_box_rep = model._config.mlod_config.mlod_box_representation

    if mlod_box_rep in ['box_3d', 'box_4ca']:
        # Boxes with angle vector output
        losses_output = _build_cls_off_ang_loss(model, prediction_dict)

    elif mlod_box_rep in ['box_8c', 'box_8co', 'box_4c']:
        losses_output = _build_cls_off_loss(model, prediction_dict)

    else:
        raise ValueError('Invalid box representation', mlod_box_rep)

    return losses_output


def _get_cls_loss(model, cls_logits, cls_gt):
    """Calculates cross entropy loss for classification

    Args:
        model: network model
        cls_logits: predicted classification logits
        cls_gt: ground truth one-hot classification vector

    Returns:
        cls_loss: cross-entropy classification loss
    """

    # Cross-entropy loss for classification
    weighted_softmax_classification_loss = \
        losses.WeightedSoftmaxLoss()
    cls_loss_weight = model._config.loss_config.cls_loss_weight
    cls_loss = weighted_softmax_classification_loss(
        cls_logits, cls_gt, weight=cls_loss_weight)

    # Normalize by the size of the minibatch
   
    with tf.variable_scope('cls_norm'):
        pos_sample = tf.not_equal(cls_gt[:,0], 0)
        pos_num = tf.reduce_sum(tf.cast(pos_sample,dtype=tf.float32))
	
        gt_pos = tf.greater(pos_num, 0.0)
        cls_loss = tf.cond(
                     gt_pos,
                     lambda: cls_loss / pos_num,
                     lambda: tf.constant(0.0))

    # Add summary scalar during training
    if model._train_val_test == 'train':
        tf.summary.scalar('classification', cls_loss)

    return cls_loss


def _get_positive_mask(positive_selection, cls_softmax, cls_gt):
    """Gets the positive mask based on the ground truth box classifications

    Args:
        positive_selection: positive selection method
            (e.g. 'corr_cls', 'not_bkg')
        cls_softmax: prediction classification softmax scores
        cls_gt: ground truth classification one-hot vector

    Returns:
        positive_mask: positive mask
    """

    # Get argmax for predicted class
    classification_argmax = tf.argmax(cls_softmax, axis=1)

    # Get the ground truth class indices back from one_hot vector
    class_indices_gt = tf.argmax(cls_gt, axis=1)

    # Mask for which predictions are not background
    not_background_mask = tf.greater(class_indices_gt, 0)

    # Combine the masks
    if positive_selection == 'corr_cls':
        # Which prediction classifications match ground truth
        correct_classifications_mask = tf.equal(
            classification_argmax, class_indices_gt)
        positive_mask = tf.logical_and(
            correct_classifications_mask, not_background_mask)
    elif positive_selection == 'not_bkg':
        positive_mask = not_background_mask
    else:
        raise ValueError('Invalid positive selection', positive_selection)

    return positive_mask


def _get_off_ang_loss(model, offsets, offsets_gt,
                      angle_vectors, angle_vectors_gt,
                      cls_softmax, cls_gt):
    """Calculates the smooth L1 combined offset and angle loss, normalized by
        the number of positives

    Args:
        model: network model
        offsets: prediction offsets
        offsets_gt: ground truth offsets
        angle_vectors: prediction angle vectors
        angle_vectors_gt: ground truth angle vectors
        cls_softmax: prediction classification softmax scores
        cls_gt: classification ground truth one-hot vector

    Returns:
        final_reg_loss: combined offset and angle vector loss
        offset_loss_norm: normalized offset loss
        ang_loss_norm: normalized angle vector loss
    """

    weighted_smooth_l1_localization_loss = losses.WeightedSmoothL1Loss()

    reg_loss_weight = model._config.loss_config.reg_loss_weight
    ang_loss_weight = model._config.loss_config.ang_loss_weight

    anchorwise_localization_loss = weighted_smooth_l1_localization_loss(
        offsets, offsets_gt, weight=reg_loss_weight)
    anchorwise_orientation_loss = weighted_smooth_l1_localization_loss(
        angle_vectors, angle_vectors_gt, weight=ang_loss_weight)

    positive_mask = _get_positive_mask(model._positive_selection,
                                       cls_softmax, cls_gt)

    # Cast to float to get number of positives
    pos_classification_floats = tf.cast(
        positive_mask, tf.float32)

    # Apply mask to only keep regression loss for positive predictions
    pos_localization_loss = tf.reduce_sum(tf.boolean_mask(
        anchorwise_localization_loss, positive_mask))
    pos_orientation_loss = tf.reduce_sum(tf.boolean_mask(
        anchorwise_orientation_loss, positive_mask))

    # Combine regression losses
    combined_reg_loss = pos_localization_loss + pos_orientation_loss

    with tf.variable_scope('reg_norm'):
        # Normalize by the number of positive/desired classes
        # only if we have any positives
        num_positives = tf.reduce_sum(pos_classification_floats)
        pos_div_cond = tf.not_equal(num_positives, 0)

        offset_loss_norm = tf.cond(
            pos_div_cond,
            lambda: pos_localization_loss / num_positives,
            lambda: tf.constant(0.0))

        ang_loss_norm = tf.cond(
            pos_div_cond,
            lambda: pos_orientation_loss / num_positives,
            lambda: tf.constant(0.0))

        final_reg_loss = tf.cond(
            pos_div_cond,
            lambda: combined_reg_loss / num_positives,
            lambda: tf.constant(0.0))

    # Add summary scalars
    if model._train_val_test == 'train':
        tf.summary.scalar('localization', offset_loss_norm)
        tf.summary.scalar('orientation', ang_loss_norm)
        tf.summary.scalar('regression_total', final_reg_loss)

        tf.summary.scalar('mb_num_positives', num_positives)

    return final_reg_loss, offset_loss_norm, ang_loss_norm


def _get_offset_only_loss(model, offsets, offsets_gt,
                          cls_softmax, cls_gt):
    """Calculates the smooth L1 combined offset and angle loss, normalized by
        the number of positives

    Args:
        model: network model
        offsets: prediction offsets
        offsets_gt: ground truth offsets
        cls_softmax: prediction classification softmax scores
        cls_gt: classification ground truth one-hot vector

    Returns:
        final_reg_loss: normalized offset loss
        offset_loss_norm: normalized offset loss
    """

    weighted_smooth_l1_loss = losses.WeightedSmoothL1Loss()

    reg_loss_weight = model._config.loss_config.reg_loss_weight
    anchorwise_localization_loss = weighted_smooth_l1_loss(
        offsets, offsets_gt, weight=reg_loss_weight)

    classification_argmax = tf.argmax(cls_softmax, axis=1)
    # Get the ground truth class indices back from one_hot vector
    class_indices_gt = tf.argmax(cls_gt, axis=1)

    # Mask for which predictions are not background
    not_background_mask = tf.greater(class_indices_gt, 0)

    # Combine the masks
    if model._positive_selection == 'corr_cls':
        # Which prediction classifications match ground truth
        correct_classifications_mask = tf.equal(
            classification_argmax,
            class_indices_gt)
        pos_classification_mask = tf.logical_and(
            correct_classifications_mask, not_background_mask)
    elif model._positive_selection == 'not_bkg':
        pos_classification_mask = not_background_mask
    else:
        raise ValueError('Invalid positive selection',
                         model._positive_selection)

    # Cast to float to get number of positives
    pos_classification_floats = tf.cast(
        pos_classification_mask, tf.float32)

    # Apply mask to only keep regression loss for positive predictions
    pos_localization_loss = tf.reduce_sum(tf.boolean_mask(
        anchorwise_localization_loss, pos_classification_mask))

    with tf.variable_scope('reg_norm'):
        # normalize by the number of positive/desired classes
        # only if we have any positives
        num_positives = tf.reduce_sum(pos_classification_floats)
        pos_div_cond = tf.not_equal(num_positives, 0)

        offset_loss_norm = tf.cond(
            pos_div_cond,
            lambda: pos_localization_loss / num_positives,
            lambda: tf.constant(0.0))

        reg_loss = tf.cond(
            pos_div_cond,
            lambda: pos_localization_loss / num_positives,
            lambda: tf.constant(0.0))

    if model._train_val_test == 'train':
        tf.summary.scalar('localization', offset_loss_norm)
        tf.summary.scalar('regression_total', reg_loss)
        tf.summary.scalar('mb_num_positives', num_positives)

    return reg_loss, offset_loss_norm


def _build_cls_off_ang_loss(model, prediction_dict):
    """Builds classification, offset, and angle vector losses.

    Args:
        model: network model
        prediction_dict: prediction dictionary

    Returns:
        losses_output: losses dictionary
    """

    # Minibatch Predictions
    mb_cls_logits = prediction_dict[model.PRED_MB_CLASSIFICATION_LOGITS]
    mb_cls_softmax = prediction_dict[model.PRED_MB_CLASSIFICATION_SOFTMAX]
    sub_mb_cls_logtis_list = prediction_dict[model.PRED_SUB_MB_CLASSIFICATION_LOGITS_LIST]
    sub_mb_cls_softmax_list = prediction_dict[model.PRED_TOP_SUB_CLASSIFICATION_SOFTMAX_LIST]
    bev_freeze_mask = prediction_dict['BEV_freeze_mask']
    sub_mb_reg_offset_list = prediction_dict[model.PRED_MB_SUB_OFFSETS_LIST]
    mb_offsets = prediction_dict[model.PRED_MB_OFFSETS]
    mb_angle_vectors = prediction_dict[model.PRED_MB_ANGLE_VECTORS]

    # Ground Truth
    mb_cls_gt = prediction_dict[model.PRED_MB_CLASSIFICATIONS_GT]
    mb_img_cls_gt = prediction_dict[model.PRED_MB_IMG_CLASSIFICATIONS_GT]
    mb_offsets_gt = prediction_dict[model.PRED_MB_OFFSETS_GT]
    mb_offsets_2d_gt = prediction_dict[model.PRED_MB_OFFSETS_2D_GT]
    mb_orientations_gt = prediction_dict[model.PRED_MB_ORIENTATIONS_GT]

    img_idx = model.img_idx

    # Decode ground truth orientations
    with tf.variable_scope('mlod_gt_angle_vectors'):
        mb_angle_vectors_gt = \
            orientation_encoder.tf_orientation_to_angle_vector(
                mb_orientations_gt)

    # Losses
    with tf.variable_scope('mlod_losses'):
        with tf.variable_scope('classification'):
            cls_loss = _get_cls_loss(model, mb_cls_logits, mb_cls_gt)
        with tf.variable_scope('sub_classification'):
            total_sub_cls_loss = 0
            for sub_idx, sub_logits in enumerate(sub_mb_cls_logtis_list):
                if sub_idx in img_idx:
                    total_sub_cls_loss += _get_cls_loss(model, sub_logits, mb_img_cls_gt)
                else:
                    total_sub_cls_loss += bev_freeze_mask*_get_cls_loss(model, sub_logits, mb_cls_gt)

        with tf.variable_scope('regression'):

            final_reg_loss, offset_loss_norm, ang_loss_norm = _get_off_ang_loss(
                model, mb_offsets, mb_offsets_gt,
                mb_angle_vectors, mb_angle_vectors_gt,
                mb_cls_softmax, mb_cls_gt)

            offset_or_ratio_list = sub_mb_reg_offset_list
            offsets_or_ratio_gt = mb_offsets_gt
            offsets_or_ratio_2d_gt = mb_offsets_2d_gt

            total_sub_reg_loss = 0
            for sub_idx,mb_sub_offsets in enumerate(offset_or_ratio_list):
                sub_mb_cls_softmax = sub_mb_cls_softmax_list[sub_idx]
                if sub_idx in img_idx:
                    reg_loss, _ = _get_offset_only_loss(model, mb_sub_offsets, offsets_or_ratio_2d_gt,
                                                sub_mb_cls_softmax, mb_img_cls_gt)
                    total_sub_reg_loss += reg_loss
                else:
                    reg_loss, _ = _get_offset_only_loss(model, mb_sub_offsets, offsets_or_ratio_gt,
                                                sub_mb_cls_softmax, mb_cls_gt)
                    total_sub_reg_loss += bev_freeze_mask*reg_loss

        with tf.variable_scope('mlod_loss'):
            sub_cls_loss_weight = model._config.loss_config.sub_cls_loss_weight
            sub_reg_loss_weight = model._config.loss_config.sub_reg_loss_weight
            mlod_loss = cls_loss + final_reg_loss + sub_cls_loss_weight*total_sub_cls_loss + sub_reg_loss_weight*total_sub_reg_loss
            tf.summary.scalar('mlod_loss', mlod_loss)

    # Loss dictionary
    losses_output = dict()

    losses_output[KEY_CLASSIFICATION_LOSS] = cls_loss
    losses_output[KEY_REGRESSION_LOSS] = final_reg_loss
    losses_output[KEY_SUB_CLASSIFICATION_LOSS] = total_sub_cls_loss
    losses_output[KEY_MLOD_LOSS] = mlod_loss

    # Separate losses for plotting
    losses_output[KEY_OFFSET_LOSS_NORM] = offset_loss_norm
    losses_output[KEY_ANG_LOSS_NORM] = ang_loss_norm

    return losses_output


def _build_cls_off_loss(model, prediction_dict):
    """Builds classification, and offset losses.

    Args:
        model: network model
        prediction_dict: prediction dictionary

    Returns:
        losses_output: losses dictionary
    """

    # Predictions
    mb_cls_logits = prediction_dict[model.PRED_MB_CLASSIFICATION_LOGITS]
    mb_cls_softmax = prediction_dict[model.PRED_MB_CLASSIFICATION_SOFTMAX]
    mb_offsets = prediction_dict[model.PRED_MB_OFFSETS]

    # Ground truth
    mb_cls_gt = prediction_dict[model.PRED_MB_CLASSIFICATIONS_GT]
    mb_offsets_gt = prediction_dict[model.PRED_MB_OFFSETS_GT]

    with tf.variable_scope('mlod_losses'):
        with tf.variable_scope('classification'):
            cls_loss = _get_cls_loss(model, mb_cls_logits, mb_cls_gt)

        with tf.variable_scope('regression'):
            final_reg_loss, offset_loss_norm = _get_offset_only_loss(
                model, mb_offsets, mb_offsets_gt, mb_cls_softmax, mb_cls_gt)

        with tf.variable_scope('mlod_loss'):
            mlod_loss = cls_loss + final_reg_loss
            tf.summary.scalar('mlod_loss', mlod_loss)

    losses_output = dict()

    losses_output[KEY_CLASSIFICATION_LOSS] = cls_loss
    losses_output[KEY_REGRESSION_LOSS] = final_reg_loss
    losses_output[KEY_MLOD_LOSS] = mlod_loss

    losses_output[KEY_OFFSET_LOSS_NORM] = offset_loss_norm

    return losses_output
