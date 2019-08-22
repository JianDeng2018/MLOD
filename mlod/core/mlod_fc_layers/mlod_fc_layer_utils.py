import tensorflow as tf
import tensorflow.contrib.slim as slim
from mlod.core.mlod_fc_layers.convlstm import ConvGRUCell, ConvLSTMCell

OFFSETS_OUTPUT_SIZE = {
    'box_3d': 6,
    'box_8c': 24,
    'box_8co': 24,
    'box_4c': 10,
    'box_4ca': 10,
}

ANG_VECS_OUTPUT_SIZE = {
    'box_3d': 2,
    'box_8c': 0,
    'box_8co': 0,
    'box_4c': 0,
    'box_4ca': 2,
}


def feature_fusion(fusion_method, inputs, input_weights):
    """Applies feature fusion to multiple inputs

    Args:
        fusion_method: 'mean' or 'concat'
        inputs: Input tensors of shape (batch_size, width, height, depth)
            If fusion_method is 'mean', inputs must have same dimensions.
            If fusion_method is 'concat', width and height must be the same.
        input_weights: Weight of each input if using 'mean' fusion method

    Returns:
        fused_features: Features after fusion
    """

    # Feature map fusion
    with tf.variable_scope('fusion'):
        fused_features = None

        if fusion_method == 'mean':
            rois_sum = tf.reduce_sum(inputs, axis=0)
            rois_mean = tf.divide(rois_sum, tf.reduce_sum(input_weights))
            fused_features = rois_mean

        elif fusion_method == 'concat':
            # Concatenate along last axis
            last_axis = len(inputs[0].get_shape()) - 1
            fused_features = tf.concat([inputs[0]*input_weights[0],inputs[1]*input_weights[1]], axis=last_axis)

        elif fusion_method == 'max':
            fused_features = tf.maximum(inputs[0], inputs[1])

        elif fusion_method == 'lstm':
            inputs_tensor = tf.stack([inputs[0]*input_weights[0],inputs[1]*input_weights[1]], axis=1)
            conv_lstm = ConvLSTMCell(
                        shape=[6, 6],
                        initializer=slim.initializers.xavier_initializer(),
                        kernel=[3,3],
                        filters=64)

            outputs, _ = tf.nn.dynamic_rnn(
                            conv_lstm,
                            inputs_tensor,
                            parallel_iterations=64,
                            dtype=tf.float32,
                            time_major=False)

            fused_features = outputs

        elif fusion_method == 'gru':
            inputs_tensor = tf.stack([inputs[0]*input_weights[0],inputs[1]*input_weights[1]], axis=1)
            conv_gru = ConvGRUCell(
                        shape=[6, 6],
                        initializer=slim.initializers.xavier_initializer(),
                        kernel=[3,3],
                        filters=64)

            outputs, _ = tf.nn.dynamic_rnn(
                            conv_gru,
                            inputs_tensor,
                            parallel_iterations=64,
                            dtype=tf.float32,
                            time_major=False)

            fused_features = outputs

        else:
            raise ValueError('Invalid fusion method', fusion_method)

    return fused_features

def scale_selection(selected_idx, sub_scale_cls_logits_tensor, scales_outputs):

    # select the suitable branches
    if selected_idx is None:
      sub_cls_softmax = tf.nn.softmax(sub_scale_cls_logits_tensor)
      
      #select the strongest signal
      max_scales_softmax = tf.reduce_max(sub_cls_softmax[:,:,1:], axis=-1)
      max_scales_softmax_indices = tf.argmax(max_scales_softmax,axis=0)

      # Create a index [[range, index]] for gather_nd 
      max_scale_softmax_indices = tf.expand_dims(max_scales_softmax_indices,axis=1)
      batch_size= tf.cast(tf.shape(max_scale_softmax_indices)[0],dtype=tf.int64)
      range_tf = tf.range(batch_size,dtype=tf.int64)
      range_tf = tf.expand_dims(range_tf, axis=1)
      selected_idx = tf.concat([max_scale_softmax_indices, range_tf],axis=1)

    scales_output_tensor = tf.stack(scales_outputs,axis=0)
    selected_scale = tf.gather_nd(scales_output_tensor,selected_idx)

    return selected_idx, selected_scale
