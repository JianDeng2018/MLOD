import numpy as np

import tensorflow as tf

from mlod.builders import mlod_fc_layers_builder
from mlod.builders import mlod_loss_builder
from mlod.core import anchor_projector
from mlod.core import anchor_encoder
from mlod.core import box_3d_encoder
from mlod.core import box_8c_encoder
from mlod.core import box_4c_encoder
from mlod.core import box_list
from mlod.core import box_list_ops
from mlod.core import model
from mlod.core import orientation_encoder
from mlod.core.models.rpn_model import RpnModel
from mlod.core.models.occlusion_mask_layer import OccMaskLayer
from mlod.core.mlod_fc_layers import mlod_fc_layer_utils
slim = tf.contrib.slim


class MlodModel(model.DetectionModel):
    ##############################
    # Keys for Predictions
    ##############################
    # Mini batch (mb) ground truth
    PRED_MB_CLASSIFICATIONS_GT = 'mlod_mb_classifications_gt'
    PRED_MB_IMG_CLASSIFICATIONS_GT = 'mlod_gt_mb_classifications_gt'
    PRED_MB_OFFSETS_GT = 'mlod_mb_offsets_gt'
    PRED_MB_OFFSETS_RATIO_GT = 'mlod_mb_offsets_ratio_gt'
    PRED_MB_OFFSETS_2D_GT = 'mlod_mb_2d_offsets'
    PRED_MB_OFFSETS_RATIO_2D_GT = 'mlod_mb_2d_offsets_ratio'
    PRED_MB_ORIENTATIONS_GT = 'mlod_mb_orientations_gt'

    # Mini batch (mb) predictions
    PRED_MB_CLASSIFICATION_LOGITS = 'mlod_mb_classification_logits'
    PRED_SUB_MB_CLASSIFICATION_LOGITS_LIST = 'mlod_sub_mb_classification_logits_list'
    PRED_MB_CLASSIFICATION_SOFTMAX = 'mlod_mb_classification_softmax'
    PRED_MB_SUB_CLASSIFICATION_SOFTMAX_LIST = 'mlod_mb_sub_classification_softmax_list'
    PRED_MB_OFFSETS = 'mlod_mb_offsets'
    PRED_MB_SUB_OFFSETS_LIST = 'mlod_mb_offsets_list'
    PRED_MB_OFFSETS_RATIO = 'mlod_mb_offsets_ratio'
    PRED_MB_SUB_OFFSETS_RATIO_LIST = 'mlod_mb_offsets_ratio_list'
    PRED_MB_ANGLE_VECTORS = 'mlod_mb_angle_vectors'

    # Top predictions after BEV NMS
    PRED_TOP_CLASSIFICATION_LOGITS = 'mlod_top_classification_logits'
    PRED_TOP_CLASSIFICATION_SOFTMAX = 'mlod_top_classification_softmax'
    PRED_TOP_SUB_CLASSIFICATION_SOFTMAX_LIST = 'mlod_top_sub_classification_softmax_list'

    PRED_TOP_PREDICTION_ANCHORS = 'mlod_top_prediction_anchors'
    PRED_TOP_PREDICTION_BOXES_3D = 'mlod_top_prediction_boxes_3d'
    PRED_TOP_ORIENTATIONS = 'mlod_top_orientations'


    # Other box representations
    PRED_TOP_BOXES_8C = 'mlod_top_regressed_boxes_8c'
    PRED_TOP_BOXES_4C = 'mlod_top_prediction_boxes_4c'

    # Mini batch (mb) predictions (for debugging)
    PRED_MB_MASK = 'mlod_mb_mask'
    PRED_MB_POS_MASK = 'mlod_mb_pos_mask'
    PRED_MB_ANCHORS_GT = 'mlod_mb_anchors_gt'
    PRED_MB_CLASS_INDICES_GT = 'mlod_mb_gt_classes'

    # All predictions (for debugging)
    PRED_ALL_CLASSIFICATIONS = 'mlod_classifications'
    PRED_ALL_OFFSETS = 'mlod_offsets'
    PRED_ALL_ANGLE_VECTORS = 'mlod_angle_vectors'

    PRED_MAX_IOUS = 'mlod_max_ious'
    PRED_ALL_IOUS = 'mlod_anchor_ious'

    ##############################
    # Keys for Loss
    ##############################
    LOSS_FINAL_CLASSIFICATION = 'mlod_classification_loss'
    LOSS_FINAL_REGRESSION = 'mlod_regression_loss'

    # (for debugging)
    LOSS_FINAL_ORIENTATION = 'mlod_orientation_loss'
    LOSS_FINAL_LOCALIZATION = 'mlod_localization_loss'

    def __init__(self, model_config, train_val_test, dataset):
        """
        Args:
            model_config: configuration for the model
            train_val_test: "train", "val", or "test"
            dataset: the dataset that will provide samples and ground truth
        """

        # Sets model configs (_config)
        super(MlodModel, self).__init__(model_config)

        self.dataset = dataset

        # Dataset config
        self._num_final_classes = self.dataset.num_classes + 1

        # Input config
        input_config = self._config.input_config
        self._bev_pixel_size = np.asarray([input_config.bev_dims_h,
                                           input_config.bev_dims_w])
        self._bev_depth = input_config.bev_depth

        self._img_pixel_size = np.asarray([input_config.img_dims_h,
                                           input_config.img_dims_w])
        self._img_depth = [input_config.img_depth]

        # MLOD config
        mlod_config = self._config.mlod_config
        self._proposal_roi_crop_size = \
            [mlod_config.mlod_proposal_roi_crop_size] * 2
        self._positive_selection = mlod_config.mlod_positive_selection
        self._nms_size = mlod_config.mlod_nms_size
        self._nms_iou_threshold = mlod_config.mlod_nms_iou_thresh
        self._path_drop_probabilities = self._config.path_drop_probabilities
        self._box_rep = mlod_config.mlod_box_representation
        self.apply_occ_mask = mlod_config.apply_occ_mask
        self.occ_quantile_level = mlod_config.occ_quantile_level

        if self._box_rep not in ['box_3d', 'box_8c', 'box_8co',
                                 'box_4c', 'box_4ca']:
            raise ValueError('Invalid box representation', self._box_rep)

        # Create the RpnModel
        self._rpn_model = RpnModel(model_config, train_val_test, dataset)

        self.lidar_only = self._rpn_model.lidar_only
        self.num_views = self._rpn_model.num_views
        self.offsets_ratio = mlod_config.offsets_ratio

        #Create the occlusion masking model
        self._n_split = 6
        self._occ_mask_layer = OccMaskLayer()

        if train_val_test not in ["train", "val", "test"]:
            raise ValueError('Invalid train_val_test value,'
                             'should be one of ["train", "val", "test"]')
        self._train_val_test = train_val_test
        self._is_training = (self._train_val_test == 'train')

        self.sample_info = {}

        self.img_idx = self.model_config.image_channel_indices
        if len(self.img_idx) > 1:
            self.multi_scale_image = True
        else:
            self.multi_scale_image = False
        self.feature_names = self.model_config.image_layer_extract
        self.feature_names = ['vgg_16/pyramid_fusion3', 'vgg_16/pyramid_fusion1']
        self.off_out_size = mlod_fc_layer_utils.OFFSETS_OUTPUT_SIZE[self._box_rep]

    def build(self):
        rpn_model = self._rpn_model

        # Share the same prediction dict as RPN
        prediction_dict = rpn_model.build()

        top_anchors = prediction_dict[RpnModel.PRED_TOP_ANCHORS]
        ground_plane = rpn_model.placeholders[RpnModel.PL_GROUND_PLANE]

        class_labels = rpn_model.placeholders[RpnModel.PL_LABEL_CLASSES]

        depth_map = prediction_dict[RpnModel.PRED_DEPTH_MAP]

        with tf.variable_scope('mlod_projection'):

            if self._config.expand_proposals_xz > 0.0:

                expand_length = self._config.expand_proposals_xz

                # Expand anchors along x and z
                with tf.variable_scope('expand_xz'):
                    expanded_dim_x = top_anchors[:, 3] + expand_length
                    expanded_dim_z = top_anchors[:, 5] + expand_length

                    expanded_anchors = tf.stack([
                        top_anchors[:, 0],
                        top_anchors[:, 1],
                        top_anchors[:, 2],
                        expanded_dim_x,
                        top_anchors[:, 4],
                        expanded_dim_z
                    ], axis=1)

                mlod_projection_in = expanded_anchors

            else:
                mlod_projection_in = top_anchors

            with tf.variable_scope('bev'):
                # Project top anchors into bev and image spaces
                bev_proposal_boxes, bev_proposal_boxes_norm = \
                    anchor_projector.project_to_bev(
                        mlod_projection_in,
                        self.dataset.kitti_utils.bev_extents)

                # Reorder projected boxes into [y1, x1, y2, x2]
                bev_proposal_boxes_tf_order = \
                    anchor_projector.reorder_projected_boxes(
                        bev_proposal_boxes)
                bev_proposal_boxes_norm_tf_order = \
                    anchor_projector.reorder_projected_boxes(
                        bev_proposal_boxes_norm)

            with tf.variable_scope('img'):
                if self.lidar_only:
                    image_shape = tf.cast(tf.shape(
                    rpn_model.placeholders[RpnModel.PL_IMG_INPUT])[1:3],
                    tf.float32)
                    img_proposal_boxes_norm_tf_order = []
                    for i in range(self.num_views):
                        img_proposal_boxes, img_proposal_boxes_norm = \
                        anchor_projector.tf_project_to_image_space(
                            mlod_projection_in,
                            rpn_model.placeholders[RpnModel.PL_CALIB_P2][i],
                            image_shape)
                        # Only reorder the normalized img
                        img_proposal_boxes_norm_tf_order.append(
                            anchor_projector.reorder_projected_boxes(
                                img_proposal_boxes_norm))
                else:
                    image_shape = tf.cast(tf.shape(
                    rpn_model.placeholders[RpnModel.PL_IMG_INPUT])[0:2],
                    tf.float32)
                    img_proposal_boxes, img_proposal_boxes_norm = \
                        anchor_projector.tf_project_to_image_space(
                            mlod_projection_in,
                            rpn_model.placeholders[RpnModel.PL_CALIB_P2],
                            image_shape)
                    img_proposal_boxes_tf_order = \
                        anchor_projector.reorder_projected_boxes(
                            img_proposal_boxes)
                    img_proposal_boxes_norm_tf_order = \
                        anchor_projector.reorder_projected_boxes(
                            img_proposal_boxes_norm)
        
            bev_feature_maps = rpn_model.bev_feature_maps
            img_feature_maps = rpn_model.img_feature_maps

        if not (self._path_drop_probabilities[0] ==
                self._path_drop_probabilities[1] == 1.0) \
           and (self.lidar_only and self.num_views > 0):

            with tf.variable_scope('mlod_path_drop'):

                img_mask = rpn_model.img_path_drop_mask
                bev_mask = rpn_model.bev_path_drop_mask
                if self.lidar_only:
                    #img_feature_maps_list = []
                    for i in range(self.num_views):
                        img_feature_maps[i] = tf.multiply(img_feature_maps[i],img_mask)
                        #img_feature_maps = img_feature_maps_list
                else:
                    img_feature_maps = tf.multiply(img_feature_maps,
                                                   img_mask)

                bev_feature_maps = tf.multiply(bev_feature_maps,
                                               bev_mask)
        else:
            bev_mask = tf.constant(1.0)
            img_mask = tf.constant(1.0)

        # ROI Pooling
        with tf.variable_scope('mlod_roi_pooling'):
            def get_box_indices(boxes):
                proposals_shape = boxes.get_shape().as_list()
                if any(dim is None for dim in proposals_shape):
                    proposals_shape = tf.shape(boxes)
                ones_mat = tf.ones(proposals_shape[:2], dtype=tf.int32)
                multiplier = tf.expand_dims(
                    tf.range(start=0, limit=proposals_shape[0]), 1)
                return tf.reshape(ones_mat * multiplier, [-1])

            bev_boxes_norm_batches = tf.expand_dims(
                bev_proposal_boxes_norm, axis=0)

            # These should be all 0's since there is only 1 image
            tf_box_indices = get_box_indices(bev_boxes_norm_batches)

            # Do ROI Pooling on BEV
            bev_rois = tf.image.crop_and_resize(
                bev_feature_maps,
                bev_proposal_boxes_norm_tf_order,
                tf_box_indices,
                self._proposal_roi_crop_size,
                name='bev_rois')
            # Do ROI Pooling on image
            if self.lidar_only:
                img_rois = []
                for i in range(self.num_views):
                    img_rois.append(
                        tf.image.crop_and_resize(
                            img_feature_maps[i],
                            img_proposal_boxes_norm_tf_order[i],
                            tf_box_indices,
                            self._proposal_roi_crop_size,
                            name='img_rois'))
            else:
                img_rois_list = []
                  
                if self.multi_scale_image:
                    img_end_points = rpn_model.img_end_points
                    #print(img_end_points)
                    img_feature_maps_list = [img_end_points[tensor_name] for tensor_name in self.feature_names]
                else:
                    two_features = False
                    if two_features:
                      img_end_points = rpn_model.img_end_points
                      img_feature_maps_list = [img_end_points[tensor_name] for tensor_name in self.feature_names]
                    else:
                      img_feature_maps_list = [img_feature_maps]

                for img_feature_maps in img_feature_maps_list:
                    img_rois_list.append(tf.image.crop_and_resize(
                            img_feature_maps,
                            img_proposal_boxes_norm_tf_order,
                            tf_box_indices,
                            self._proposal_roi_crop_size,
                            name='img_rois'))

        # Occlusion masking
        if self.apply_occ_mask:
            ref_depth_min = mlod_projection_in[:,2]-mlod_projection_in[:,5]/2-0.5
            ref_depth_max = mlod_projection_in[:,2]+mlod_projection_in[:,5]/2+0.5
            #no background masking
            #ref_depth_max = tf.ones_like(ref_depth_min)*100

            occ_mask = self._occ_mask_layer.build(
                depth_map,
                img_proposal_boxes_norm_tf_order,
                tf_box_indices,
                ref_depth_min,
                ref_depth_max,
                self._n_split,
                [8,8],
                self._proposal_roi_crop_size,
                self.occ_quantile_level)

            img_rois_masked_list = [tf.multiply(img_rois,occ_mask,name='masked_img') for img_rois in img_rois_list]
        else:
            img_rois_masked_list = img_rois_list

        # Get anchors dimension    
        boxes_3d_x_dim = tf.abs(bev_proposal_boxes[:,0]-bev_proposal_boxes[:,2])
        boxes_3d_z_dim = tf.abs(bev_proposal_boxes[:,1]-bev_proposal_boxes[:,3])
        boxes_3d_dim = tf.stack([boxes_3d_x_dim,boxes_3d_x_dim,boxes_3d_x_dim,boxes_3d_x_dim,
                                 boxes_3d_z_dim,boxes_3d_z_dim,boxes_3d_z_dim,boxes_3d_z_dim,
                                 tf.ones_like(boxes_3d_z_dim),tf.ones_like(boxes_3d_z_dim)],axis=1)

        boxes_2d_x_dim = tf.abs(img_proposal_boxes[:,0]-img_proposal_boxes[:,2])
        boxes_2d_y_dim = tf.abs(img_proposal_boxes[:,1]-img_proposal_boxes[:,3])
        boxes_2d_dim = tf.stack([boxes_2d_x_dim,boxes_2d_y_dim,
                                 tf.ones_like(boxes_2d_x_dim),
                                 tf.ones_like(boxes_2d_y_dim)],axis=1)

        # Fully connected layers (Box Predictor)
        mlod_layers_config = self.model_config.layers_config.mlod_config
        cls_input_weights = [mlod_layers_config.cls_input_weights[0]*bev_mask, mlod_layers_config.cls_input_weights[1]*img_mask]
        reg_input_weights = [mlod_layers_config.reg_input_weights[0]*bev_mask, mlod_layers_config.reg_input_weights[1]*img_mask]

        multi_check = rpn_model.multi_check
        cls_reg_separated = rpn_model.cls_reg_separated
        reg_var = self._config.reg_var


        if cls_reg_separated:
            fusion_cls_out = ['cls']
            fusion_reg_out = ['offset','ang']
        else:
            fusion_net_out = ['cls','offset','ang']

        if self.lidar_only:
            img_rois_masked.append(bev_rois)
            mlod_mask = [img_mask]*self.num_views+[bev_mask]
            fc_output_layers = \
                mlod_fc_layers_builder.build(
                    layers_config=mlod_layers_config,
                    input_rois=img_rois_masked,
                    input_weights=mlod_mask,
                    num_final_classes=self._num_final_classes,
                    box_rep=self._box_rep,
                    top_anchors=top_anchors,
                    ground_plane=ground_plane,
                    is_training=self._is_training,
                    cls_reg_separated=cls_reg_separated)
        else:
            if two_features:
              rois_masked_list_cls = [img_rois_masked_list[0]]
              rois_masked_list_reg = [img_rois_masked_list[1]]
              rois_masked_list_cls.insert(0,bev_rois)
              rois_masked_list_reg.insert(0,bev_rois)
            else:
              rois_masked_list = img_rois_masked_list
              rois_masked_list.insert(0,bev_rois)
              rois_masked_list_cls = rois_masked_list
              rois_masked_list_reg = rois_masked_list
            if not cls_reg_separated:    
                fc_output_layers = \
                    mlod_fc_layers_builder.build(
                        layers_config=mlod_layers_config,
                        input_rois=rois_masked_list,
                        cls_input_weights=cls_input_weights,
                        reg_input_weights=reg_input_weights,
                        num_final_classes=self._num_final_classes,
                        box_rep=self._box_rep,
                        top_anchors=top_anchors,
                        ground_plane=ground_plane,
                        is_training=self._is_training,
                        variables_name='box_classifier_regressor',
                        multi_check=multi_check,
                        net_out = fusion_net_out,
                        img_idx = self.img_idx)

                all_cls_logits = \
                    fc_output_layers[mlod_fc_layers_builder.KEY_CLS_LOGITS]
                all_offsets = fc_output_layers[mlod_fc_layers_builder.KEY_OFFSETS]
                all_angle_vectors = \
                    fc_output_layers.get(mlod_fc_layers_builder.KEY_ANGLE_VECTORS)
                if multi_check:
                    sub_cls_logits_list = \
                        fc_output_layers[mlod_fc_layers_builder.KEY_SUB_CLS_LOGITS_LIST]
                    sub_reg_offset_list = fc_output_layers[mlod_fc_layers_builder.KEY_SUB_REG_OFFSETS_LIST]
                else:
                    sub_cls_logits_list = []
                    sub_reg_offset_list = []

            else:            
                 
                fc_output_layers1 = \
                    mlod_fc_layers_builder.build(
                        layers_config=mlod_layers_config,
                        input_rois=rois_masked_list_cls,
                        cls_input_weights=cls_input_weights,
                        reg_input_weights=reg_input_weights,
                        num_final_classes=self._num_final_classes,
                        box_rep=self._box_rep,
                        top_anchors=top_anchors,
                        ground_plane=ground_plane,
                        is_training=self._is_training,
                        variables_name='box_classifier',
                        multi_check=multi_check,
                        net_out = fusion_cls_out,
                        img_idx = self.img_idx)

                all_cls_logits = \
                   fc_output_layers1[mlod_fc_layers_builder.KEY_CLS_LOGITS]
                sub_cls_logits_list = \
                    fc_output_layers1[mlod_fc_layers_builder.KEY_SUB_CLS_LOGITS_LIST]
                selected_img_idx = fc_output_layers1[mlod_fc_layers_builder.KEY_SELECTED_IMG_IDX]

                fc_output_layers2 = \
                    mlod_fc_layers_builder.build(
                        layers_config=mlod_layers_config,
                        input_rois=rois_masked_list_reg,
                        cls_input_weights=cls_input_weights,
                        reg_input_weights=reg_input_weights,
                        num_final_classes=self._num_final_classes,
                        box_rep=self._box_rep,
                        top_anchors=top_anchors,
                        ground_plane=ground_plane,
                        is_training=self._is_training,
                        variables_name='box_regressor',
                        multi_check=multi_check,
                        net_out = fusion_reg_out,
                        img_idx = self.img_idx,
                        selected_img_idx_in = selected_img_idx)

                if self.offsets_ratio:
                    all_offsets_ratio = fc_output_layers2[mlod_fc_layers_builder.KEY_OFFSETS]
                    sub_reg_offset_ratio_list = fc_output_layers2[mlod_fc_layers_builder.KEY_SUB_REG_OFFSETS_LIST]
                    
                    all_offsets = all_offsets_ratio*boxes_3d_dim*reg_var
                    sub_reg_offset_list = []
                    for branch_idx, sub_reg_offset_ratio in enumerate(sub_reg_offset_ratio_list):
                        if branch_idx in self.img_idx :
                            sub_reg_offset = sub_reg_offset_ratio*boxes_2d_dim*reg_var
                        else:
                            sub_reg_offset = sub_reg_offset_ratio*boxes_3d_dim*reg_var
                        sub_reg_offset_list.append(sub_reg_offset)
                else:
                    all_offsets_multi_classes = fc_output_layers2[mlod_fc_layers_builder.KEY_OFFSETS]
                    sub_reg_offset_multi_classes_list = fc_output_layers2[mlod_fc_layers_builder.KEY_SUB_REG_OFFSETS_LIST]

                all_angle_vectors = \
                    fc_output_layers2.get(mlod_fc_layers_builder.KEY_ANGLE_VECTORS)

        #select class in offsets
        print(all_offsets_multi_classes, all_cls_logits)
        all_offsets = self.class_selection(all_offsets_multi_classes, all_cls_logits, self.off_out_size) 
        sub_reg_offset_list = []
        for branch_idx, sub_offsets_mc in enumerate(sub_reg_offset_multi_classes_list):
            sub_logits = sub_cls_logits_list[branch_idx]
            if branch_idx in self.img_idx:
                sub_offsets = self.class_selection(sub_offsets_mc, sub_logits, 4) 
            else:
                sub_offsets = self.class_selection(sub_offsets_mc, sub_logits, self.off_out_size) 
            sub_reg_offset_list.append(sub_offsets)

        sub_cls_softmax_list = []
        with tf.variable_scope('softmax'):
            all_cls_softmax = tf.nn.softmax(
                all_cls_logits)

            for sub_logits in sub_cls_logits_list:
                sub_cls_softmax = tf.nn.softmax(sub_logits)
                sub_cls_softmax_list.append(sub_cls_softmax)

        ######################################################
        # Subsample mini_batch for the loss function
        ######################################################
        # Get the ground truth tensors
        anchors_gt = rpn_model.placeholders[RpnModel.PL_LABEL_ANCHORS]
        if self._box_rep in ['box_3d', 'box_4ca']:
            boxes_3d_gt = rpn_model.placeholders[RpnModel.PL_LABEL_BOXES_3D]
            orientations_gt = boxes_3d_gt[:, 6]
        elif self._box_rep in ['box_8c', 'box_8co', 'box_4c']:
            boxes_3d_gt = rpn_model.placeholders[RpnModel.PL_LABEL_BOXES_3D]
        else:
            raise NotImplementedError('Ground truth tensors not implemented')
        boxes_2d_gt = rpn_model.placeholders[RpnModel.PL_LABEL_BOXES_2D]

        # Project anchor_gts to 2D bev
        with tf.variable_scope('mlod_gt_projection'):
            # TODO: (#140) fix kitti_util
            bev_anchor_boxes_gt, _ = anchor_projector.project_to_bev(
                anchors_gt, self.dataset.kitti_utils.bev_extents)

            bev_anchor_boxes_gt_tf_order = \
                anchor_projector.reorder_projected_boxes(bev_anchor_boxes_gt)

            img_anchor_boxes_gt_tf_order = \
                anchor_projector.reorder_projected_boxes(boxes_2d_gt)

        with tf.variable_scope('mlod_box_list'):
            #bev
            # Convert to box_list format
            anchor_box_list_gt = box_list.BoxList(bev_anchor_boxes_gt_tf_order)
            anchor_box_list = box_list.BoxList(bev_proposal_boxes_tf_order)

            #img
            img_box_list_gt = box_list.BoxList(img_anchor_boxes_gt_tf_order)
            img_box_list = box_list.BoxList(img_proposal_boxes_tf_order)

        mb_mask, mb_bev_class_label_indices, mb_img_class_label_indices, \
            mb_gt_indices, mb_img_gt_indices = \
            self.sample_mini_batch(
                anchor_box_list_gt=anchor_box_list_gt,
                anchor_box_list=anchor_box_list,
                img_box_list_gt=img_box_list_gt,
                img_box_list=img_box_list,
                class_labels=class_labels)

        # Create classification one_hot vector
        with tf.variable_scope('mlod_one_hot_classes'):
            mb_classification_gt = tf.one_hot(
                mb_bev_class_label_indices,
                depth=self._num_final_classes,
                on_value=1.0 - self._config.label_smoothing_epsilon,
                off_value=(self._config.label_smoothing_epsilon /
                           self.dataset.num_classes))
        
        with tf.variable_scope('mlod_img_one_hot_classes'):
            mb_img_classification_gt = tf.one_hot(
                mb_img_class_label_indices,
                depth=self._num_final_classes,
                on_value=1.0 - self._config.label_smoothing_epsilon,
                off_value=(self._config.label_smoothing_epsilon /
                           self.dataset.num_classes))

        # Mask predictions
        with tf.variable_scope('mlod_apply_mb_mask'):
            # Classification
            mb_classifications_logits = tf.boolean_mask(
                all_cls_logits, mb_mask)
            mb_classifications_softmax = tf.boolean_mask(
                all_cls_softmax, mb_mask)
            sub_mb_classifications_logits_list = []
            for br_idx, sub_cls_logits in enumerate(sub_cls_logits_list):
                sub_mb_classifications_logits = tf.boolean_mask(
                    sub_cls_logits, mb_mask)
                sub_mb_classifications_logits_list.append(sub_mb_classifications_logits)
            mb_sub_classifications_softmax_list = []
            for br_idx, sub_cls_softmax in enumerate(sub_cls_softmax_list):
                mb_sub_classifications_softmax = tf.boolean_mask(
                    sub_cls_softmax, mb_mask)
                mb_sub_classifications_softmax_list.append(mb_sub_classifications_softmax)

            # Offsets
            mb_offsets = tf.boolean_mask(all_offsets, mb_mask)
     
            mb_sub_offsets_list = []
            for sub_offset in sub_reg_offset_list:
                mb_sub_offsets = tf.boolean_mask(sub_offset, mb_mask)
                mb_sub_offsets_list.append(mb_sub_offsets)

            # Angle Vectors
            if all_angle_vectors is not None:
                mb_angle_vectors = tf.boolean_mask(all_angle_vectors, mb_mask)
            else:
                mb_angle_vectors = None

        # Encode anchor offsets
        with tf.variable_scope('mlod_encode_mb_anchors'):
            mb_anchors = tf.boolean_mask(top_anchors, mb_mask)

            if self._box_rep == 'box_3d':
                # Gather corresponding ground truth anchors for each mb sample
                mb_anchors_gt = tf.gather(anchors_gt, mb_gt_indices)
                mb_offsets_gt = anchor_encoder.tf_anchor_to_offset(
                    mb_anchors, mb_anchors_gt)

                # Gather corresponding ground truth orientation for each
                # mb sample
                mb_orientations_gt = tf.gather(orientations_gt,
                                               mb_gt_indices)
            elif self._box_rep in ['box_8c', 'box_8co']:

                # Get boxes_3d ground truth mini-batch and convert to box_8c
                mb_boxes_3d_gt = tf.gather(boxes_3d_gt, mb_gt_indices)
                if self._box_rep == 'box_8c':
                    mb_boxes_8c_gt = \
                        box_8c_encoder.tf_box_3d_to_box_8c(mb_boxes_3d_gt)
                elif self._box_rep == 'box_8co':
                    mb_boxes_8c_gt = \
                        box_8c_encoder.tf_box_3d_to_box_8co(mb_boxes_3d_gt)

                # Convert proposals: anchors -> box_3d -> box8c
                proposal_boxes_3d = \
                    box_3d_encoder.anchors_to_box_3d(top_anchors, fix_lw=True)
                proposal_boxes_8c = \
                    box_8c_encoder.tf_box_3d_to_box_8c(proposal_boxes_3d)

                # Get mini batch offsets
                mb_boxes_8c = tf.boolean_mask(proposal_boxes_8c, mb_mask)
                mb_offsets_gt = box_8c_encoder.tf_box_8c_to_offsets(
                    mb_boxes_8c, mb_boxes_8c_gt)

                # Flatten the offsets to a (N x 24) vector
                mb_offsets_gt = tf.reshape(mb_offsets_gt, [-1, 24])

            elif self._box_rep in ['box_4c', 'box_4ca']:

                # Get ground plane for box_4c conversion
                ground_plane = self._rpn_model.placeholders[
                    self._rpn_model.PL_GROUND_PLANE]

                # Convert gt boxes_3d -> box_4c
                mb_boxes_3d_gt = tf.gather(boxes_3d_gt, mb_gt_indices)
                mb_boxes_4c_gt = box_4c_encoder.tf_box_3d_to_box_4c(
                    mb_boxes_3d_gt, ground_plane)

                # Convert proposals: anchors -> box_3d -> box_4c
                proposal_boxes_3d = \
                    box_3d_encoder.anchors_to_box_3d(top_anchors, fix_lw=True)
                proposal_boxes_4c = \
                    box_4c_encoder.tf_box_3d_to_box_4c(proposal_boxes_3d,
                                                       ground_plane)

                # Get mini batch
                mb_boxes_4c = tf.boolean_mask(proposal_boxes_4c, mb_mask)
                mb_offsets_gt = box_4c_encoder.tf_box_4c_to_offsets(
                    mb_boxes_4c, mb_boxes_4c_gt)

                if self._box_rep == 'box_4ca':
                    # Gather corresponding ground truth orientation for each
                    # mb sample
                    mb_orientations_gt = tf.gather(orientations_gt,
                                                   mb_gt_indices)

            else:
                raise NotImplementedError(
                    'Anchor encoding not implemented for', self._box_rep)

            #2d bounding boxes offset
            
            # anchor projected 2d bounding box
            
            #mb_img_proposal_boxes_norm = tf.boolean_mask(img_proposal_boxes_norm, mb_mask)
            #mb_boxes_2d = mb_img_proposal_boxes_norm*[image_w,image_h,image_w,image_h]
            #mb_boxes_w = mb_boxes_2d[:,2] - mb_boxes_2d[:,0]
            #mb_boxes_h = mb_boxes_2d[:,3] - mb_boxes_2d[:,1]
            mb_boxes_2d = tf.boolean_mask(img_proposal_boxes,mb_mask)
            mb_boxes_2d_gt = tf.gather(boxes_2d_gt, mb_img_gt_indices)
            mb_offsets_2d_gt = anchor_encoder.tf_2d_box_to_offset(mb_boxes_2d, mb_boxes_2d_gt)  

        ######################################################
        # ROI summary images
        ######################################################
        mlod_mini_batch_size = \
            self.dataset.kitti_utils.mini_batch_utils.mlod_mini_batch_size
        with tf.variable_scope('bev_mlod_rois'):
            mb_bev_anchors_norm = tf.boolean_mask(
                bev_proposal_boxes_norm_tf_order, mb_mask)
            mb_bev_box_indices = tf.zeros_like(mb_gt_indices, dtype=tf.int32)

            # Show the ROIs of the BEV input density map
            # for the mini batch anchors
            bev_input_rois = tf.image.crop_and_resize(
                self._rpn_model._bev_preprocessed,
                mb_bev_anchors_norm,
                mb_bev_box_indices,
                (32, 32))

            bev_input_roi_summary_images = tf.split(
                bev_input_rois, self._bev_depth, axis=3)
            tf.summary.image('bev_mlod_rois',
                             bev_input_roi_summary_images[-1],
                             max_outputs=mlod_mini_batch_size)

        with tf.variable_scope('img_mlod_rois'):
            if self.lidar_only:
                for i in range(self.num_views):
                    # ROIs on image input
                    mb_img_anchors_norm = tf.boolean_mask(
                        img_proposal_boxes_norm_tf_order[i], mb_mask)
                    mb_img_box_indices = tf.zeros_like(mb_gt_indices, dtype=tf.int32)

                    # Do test ROI pooling on mini batch
                    img_input_rois = tf.image.crop_and_resize(
                        tf.expand_dims(self._rpn_model._img_preprocessed[i],axis=0),
                        mb_img_anchors_norm,
                        mb_img_box_indices,
                        (32, 32))

                    tf.summary.image('img_mlod_rois',
                                     img_input_rois,
                                     max_outputs=mlod_mini_batch_size)
            else:
                # ROIs on image input
                mb_img_anchors_norm = tf.boolean_mask(
                    img_proposal_boxes_norm_tf_order, mb_mask)
                mb_img_box_indices = tf.zeros_like(mb_gt_indices, dtype=tf.int32)

                # Do test ROI pooling on mini batch
                img_input_rois = tf.image.crop_and_resize(
                    self._rpn_model._img_preprocessed,
                    mb_img_anchors_norm,
                    mb_img_box_indices,
                    (32, 32))

                tf.summary.image('img_mlod_rois',
                                 img_input_rois,
                                 max_outputs=mlod_mini_batch_size)


        ######################################################
        # Final Predictions
        ######################################################
        # Get orientations from angle vectors
        if all_angle_vectors is not None:
            with tf.variable_scope('mlod_orientation'):
                all_orientations = \
                    orientation_encoder.tf_angle_vector_to_orientation(
                        all_angle_vectors)

        # Apply offsets to regress proposals
        with tf.variable_scope('mlod_regression'):
            if self._box_rep == 'box_3d':
                prediction_anchors = \
                    anchor_encoder.offset_to_anchor(top_anchors,
                                                    all_offsets)

            elif self._box_rep in ['box_8c', 'box_8co']:
                # Reshape the 24-dim regressed offsets to (N x 3 x 8)
                reshaped_offsets = tf.reshape(all_offsets,
                                              [-1, 3, 8])
                # Given the offsets, get the boxes_8c
                prediction_boxes_8c = \
                    box_8c_encoder.tf_offsets_to_box_8c(proposal_boxes_8c,
                                                        reshaped_offsets)
                # Convert corners back to box3D
                prediction_boxes_3d = \
                    box_8c_encoder.box_8c_to_box_3d(prediction_boxes_8c)

                # Convert the box_3d to anchor format for nms
                prediction_anchors = \
                    box_3d_encoder.tf_box_3d_to_anchor(prediction_boxes_3d)

            elif self._box_rep in ['box_4c', 'box_4ca']:
                # Convert predictions box_4c -> box_3d
                prediction_boxes_4c = \
                    box_4c_encoder.tf_offsets_to_box_4c(proposal_boxes_4c,
                                                        all_offsets)

                prediction_boxes_3d = \
                    box_4c_encoder.tf_box_4c_to_box_3d(prediction_boxes_4c,
                                                       ground_plane)

                # Convert to anchor format for nms
                prediction_anchors = \
                    box_3d_encoder.tf_box_3d_to_anchor(prediction_boxes_3d)

            else:
                raise NotImplementedError('Regression not implemented for',
                                          self._box_rep)

        # Apply Non-oriented NMS in BEV
        with tf.variable_scope('mlod_nms'):
            bev_extents = self.dataset.kitti_utils.bev_extents

            with tf.variable_scope('bev_projection'):
                # Project predictions into BEV
                mlod_bev_boxes, _ = anchor_projector.project_to_bev(
                    prediction_anchors, bev_extents)
                mlod_bev_boxes_tf_order = \
                    anchor_projector.reorder_projected_boxes(
                        mlod_bev_boxes)

            # Get top score from second column onward
            all_top_scores = tf.reduce_max(all_cls_logits[:, 1:], axis=1)

            # Apply NMS in BEV
            nms_indices = tf.image.non_max_suppression(
                mlod_bev_boxes_tf_order,
                all_top_scores,
                max_output_size=self._nms_size,
                iou_threshold=self._nms_iou_threshold)

            # Gather predictions from NMS indices
            top_classification_logits = tf.gather(all_cls_logits,
                                                  nms_indices)
            top_classification_softmax = tf.gather(all_cls_softmax,
                                                   nms_indices)
            top_sub_classification_softmax_list = []
            for sub_cls_softmax in sub_cls_softmax_list:
                top_sub_classification_softmax_list.append(tf.gather(sub_cls_softmax,
                                                   nms_indices))
            top_2d_proposal = tf.gather(img_proposal_boxes_norm,nms_indices)

            top_prediction_anchors = tf.gather(prediction_anchors,
                                               nms_indices)

            if self._box_rep == 'box_3d':
                top_orientations = tf.gather(
                    all_orientations, nms_indices)

            elif self._box_rep in ['box_8c', 'box_8co']:
                top_prediction_boxes_3d = tf.gather(
                    prediction_boxes_3d, nms_indices)
                top_prediction_boxes_8c = tf.gather(
                    prediction_boxes_8c, nms_indices)

            elif self._box_rep == 'box_4c':
                top_prediction_boxes_3d = tf.gather(
                    prediction_boxes_3d, nms_indices)
                top_prediction_boxes_4c = tf.gather(
                    prediction_boxes_4c, nms_indices)

            elif self._box_rep == 'box_4ca':
                top_prediction_boxes_3d = tf.gather(
                    prediction_boxes_3d, nms_indices)
                top_prediction_boxes_4c = tf.gather(
                    prediction_boxes_4c, nms_indices)
                top_orientations = tf.gather(
                    all_orientations, nms_indices)

            else:
                raise NotImplementedError('NMS gather not implemented for',
                                          self._box_rep)

        if self._train_val_test in ['train', 'val']:
            # Additional entries are added to the shared prediction_dict
            # Mini batch predictions
            prediction_dict['cls_softmax'] = all_cls_softmax
            prediction_dict[self.PRED_SUB_MB_CLASSIFICATION_LOGITS_LIST] = \
                sub_mb_classifications_logits_list
            prediction_dict[self.PRED_MB_CLASSIFICATION_LOGITS] = \
                mb_classifications_logits
            prediction_dict[self.PRED_MB_CLASSIFICATION_SOFTMAX] = \
                mb_classifications_softmax
            prediction_dict[self.PRED_MB_SUB_CLASSIFICATION_SOFTMAX_LIST] = \
                mb_sub_classifications_softmax_list
            prediction_dict[self.PRED_MB_OFFSETS] = mb_offsets
            prediction_dict[self.PRED_MB_SUB_OFFSETS_LIST] = mb_sub_offsets_list

            prediction_dict['top_3d_proposal'] = top_2d_proposal

            prediction_dict['2d_gt_box_rescale'] = img_anchor_boxes_gt_tf_order
            prediction_dict['3d_gt_box_rescale'] = bev_anchor_boxes_gt_tf_order
            prediction_dict['2d_proposed_box'] = img_proposal_boxes_tf_order
            prediction_dict['3d_proposed_box'] = bev_proposal_boxes_tf_order

            # Mini batch ground truth
            prediction_dict[self.PRED_MB_CLASSIFICATIONS_GT] = \
                mb_classification_gt
            prediction_dict[self.PRED_MB_IMG_CLASSIFICATIONS_GT] = \
                mb_img_classification_gt
        
            prediction_dict[self.PRED_MB_OFFSETS_GT] = mb_offsets_gt
            prediction_dict[self.PRED_MB_OFFSETS_2D_GT] = mb_offsets_2d_gt

            # Top NMS predictions
            prediction_dict[self.PRED_TOP_CLASSIFICATION_LOGITS] = \
                top_classification_logits
            prediction_dict[self.PRED_TOP_CLASSIFICATION_SOFTMAX] = \
                top_classification_softmax
            prediction_dict[self.PRED_TOP_SUB_CLASSIFICATION_SOFTMAX_LIST] = \
                top_sub_classification_softmax_list

            prediction_dict[self.PRED_TOP_PREDICTION_ANCHORS] = \
                top_prediction_anchors

            # Mini batch predictions (for debugging)
            prediction_dict[self.PRED_MB_MASK] = mb_mask
            prediction_dict[self.PRED_MB_CLASS_INDICES_GT] = \
                mb_bev_class_label_indices
            
            prediction_dict['gt_indices'] = mb_gt_indices
            prediction_dict['img_gt_indices'] = mb_img_gt_indices
            prediction_dict['img_gt_label_indices'] = mb_img_class_label_indices

            # All predictions (for debugging)
            prediction_dict[self.PRED_ALL_CLASSIFICATIONS] = \
                all_cls_logits
            prediction_dict[self.PRED_ALL_OFFSETS] = all_offsets

            # Path drop masks (for debugging)
            prediction_dict['bev_mask'] = bev_mask
            prediction_dict['img_mask'] = img_mask
            prediction_dict['BEV_freeze_mask'] = self._rpn_model.bev_freeze_mask

        else:
            # self._train_val_test == 'test'
            prediction_dict[self.PRED_TOP_CLASSIFICATION_SOFTMAX] = \
                top_classification_softmax
            prediction_dict[self.PRED_TOP_SUB_CLASSIFICATION_SOFTMAX_LIST] = \
                top_sub_classification_softmax_list
            prediction_dict[self.PRED_TOP_PREDICTION_ANCHORS] = \
                top_prediction_anchors
            prediction_dict[self.PRED_ALL_OFFSETS] = all_offsets

        if self._box_rep == 'box_3d':
            prediction_dict[self.PRED_MB_ANCHORS_GT] = mb_anchors_gt
            prediction_dict[self.PRED_MB_ORIENTATIONS_GT] = mb_orientations_gt
            prediction_dict[self.PRED_MB_ANGLE_VECTORS] = mb_angle_vectors

            prediction_dict[self.PRED_TOP_ORIENTATIONS] = top_orientations

            # For debugging
            prediction_dict[self.PRED_ALL_ANGLE_VECTORS] = all_angle_vectors

        elif self._box_rep in ['box_8c', 'box_8co']:
            prediction_dict[self.PRED_TOP_PREDICTION_BOXES_3D] = \
                top_prediction_boxes_3d

            # Store the corners before converting for visualization purposes
            prediction_dict[self.PRED_TOP_BOXES_8C] = top_prediction_boxes_8c

        elif self._box_rep == 'box_4c':
            prediction_dict[self.PRED_TOP_PREDICTION_BOXES_3D] = \
                top_prediction_boxes_3d
            prediction_dict[self.PRED_TOP_BOXES_4C] = top_prediction_boxes_4c

        elif self._box_rep == 'box_4ca':
            if self._train_val_test in ['train', 'val']:
                prediction_dict[self.PRED_MB_ORIENTATIONS_GT] = \
                    mb_orientations_gt
                prediction_dict[self.PRED_MB_ANGLE_VECTORS] = mb_angle_vectors

            prediction_dict[self.PRED_TOP_PREDICTION_BOXES_3D] = \
                top_prediction_boxes_3d
            prediction_dict[self.PRED_TOP_BOXES_4C] = top_prediction_boxes_4c
            prediction_dict[self.PRED_TOP_ORIENTATIONS] = top_orientations

        else:
            raise NotImplementedError('Prediction dict not implemented for',
                                      self._box_rep)

        return prediction_dict

    def sample_mini_batch(self, anchor_box_list_gt, anchor_box_list,
                          img_box_list_gt,img_box_list,
                          class_labels):

        with tf.variable_scope('mlod_create_mb_mask'):
            # Get IoU for every anchor 
            all_bev_ious = box_list_ops.iou(anchor_box_list_gt, anchor_box_list)
            max_bev_ious = tf.reduce_max(all_bev_ious, axis=0)
            max_bev_iou_indices = tf.argmax(all_bev_ious, axis=0)

            all_img_ious = box_list_ops.iou(img_box_list_gt, img_box_list)
            max_img_ious = tf.reduce_max(all_img_ious, axis=0)
            max_img_iou_indices = tf.argmax(all_img_ious, axis=0)

            # Sample a pos/neg mini-batch from anchors with highest IoU match
            mini_batch_utils = self.dataset.kitti_utils.mini_batch_utils
            #bev mask 
            mb_bev_mask, mb_pos_bev_mask = mini_batch_utils.sample_mlod_mini_batch(
                    max_bev_ious)

            mb_mask, mb_pos_img_mask = mini_batch_utils.double_sample_img_mini_batch(
                    max_img_ious, mb_bev_mask)

            mb_bev_class_label_indices = mini_batch_utils.mask_class_label_indices(
                mb_pos_bev_mask, mb_mask, max_bev_iou_indices, class_labels)
            mb_img_class_label_indices = mini_batch_utils.mask_class_label_indices(
                mb_pos_img_mask, mb_mask, max_img_iou_indices, class_labels)

            mb_bev_gt_indices = tf.boolean_mask(max_bev_iou_indices, mb_mask)
            mb_img_gt_indices = tf.boolean_mask(max_img_iou_indices, mb_mask)

        return mb_mask, mb_bev_class_label_indices, mb_img_class_label_indices, mb_bev_gt_indices, mb_img_gt_indices

    def create_feed_dict(self):
        feed_dict = self._rpn_model.create_feed_dict()
        self.sample_info = self._rpn_model.sample_info
        return feed_dict

    def loss(self, prediction_dict):
        # Note: The loss should be using mini-batch values only
        loss_dict, rpn_loss = self._rpn_model.loss(prediction_dict)
        losses_output = mlod_loss_builder.build(self, prediction_dict)

        classification_loss = \
            losses_output[mlod_loss_builder.KEY_CLASSIFICATION_LOSS]

        final_reg_loss = losses_output[mlod_loss_builder.KEY_REGRESSION_LOSS]

        mlod_loss = losses_output[mlod_loss_builder.KEY_MLOD_LOSS]

        offset_loss_norm = \
            losses_output[mlod_loss_builder.KEY_OFFSET_LOSS_NORM]

        loss_dict.update({self.LOSS_FINAL_CLASSIFICATION: classification_loss})
        loss_dict.update({self.LOSS_FINAL_REGRESSION: final_reg_loss})

        # Add localization and orientation losses to loss dict for plotting
        loss_dict.update({self.LOSS_FINAL_LOCALIZATION: offset_loss_norm})

        ang_loss_loss_norm = losses_output.get(
            mlod_loss_builder.KEY_ANG_LOSS_NORM)
        if ang_loss_loss_norm is not None:
            loss_dict.update({self.LOSS_FINAL_ORIENTATION: ang_loss_loss_norm})

        with tf.variable_scope('model_total_loss'):
            total_loss = rpn_loss + mlod_loss

        return loss_dict, total_loss

    def class_selection(self,offsets,cls_logits, off_out_size):
      off_out_all = tf.reshape(offsets, [-1, self._num_final_classes,off_out_size])
      #select the regression output by class
      cls_idx = tf.argmax(cls_logits,axis=1)
      cls_idx = tf.expand_dims(cls_idx,axis=-1)
      batch_size= tf.cast(tf.shape(offsets)[0],dtype=tf.int64)
      range_tf = tf.range(batch_size,dtype=tf.int64)
      range_tf = tf.expand_dims(range_tf, axis=1)
      selected_idx = tf.concat([range_tf, cls_idx],axis=1)
      
      off_out = tf.gather_nd(off_out_all,selected_idx)

      return off_out

    def resnet(self, input_layer):
        layer_size = 128
        start_layer = slim.conv2d(input_layer,
                            layer_size,
                            [3,3],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={
                            'is_training': self._is_training})
        layer = start_layer
        for i in range(2):
            bn_features = slim.batch_norm(layer, is_training=self._is_training)
            relu_feature = tf.nn.relu(bn_features)
            layer = slim.conv2d(relu_feature,layer_size,[3,3],activation_fn=None)
        return layer + start_layer
