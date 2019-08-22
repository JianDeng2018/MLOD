import os

import cv2
import numpy as np
import tensorflow as tf
import vtk

import matplotlib.pyplot as plt

from wavedata.tools.obj_detection import obj_utils
from wavedata.tools.visualization import vis_utils
from wavedata.tools.visualization.vtk_boxes import VtkBoxes
from wavedata.tools.visualization.vtk_point_cloud import VtkPointCloud

import mlod
from mlod.builders import config_builder_util
from mlod.builders.dataset_builder import DatasetBuilder
from mlod.core import anchor_projector
from mlod.core import box_3d_encoder
from mlod.core import box_list, box_list_ops
from mlod.utils import demo_utils


COLOUR_SCHEME = {
    "Car": (255, 0, 0),  # Red
    "Pedestrian": (80, 80, 255),  # Blue
    "Cyclist": (150, 50, 100),  # Purple
    "DontCare": (255, 255, 255),  # White

    "OrthoGt": (0, 255, 0),  # Green

    "BackgroundProposal": (80, 80, 80),  # Dark Gray
    "NegativeProposal": (120, 120, 120),  # Dark Gray
    "MiddleProposal": (180, 180, 180),  # Dark Gray
    "PositiveProposal": (255, 255, 0),  # Yellow

    "Positive": (0, 255, 255),  # Teal
    "Negative": (255, 0, 255)  # Purple
}


def main():
    """
    This demo shows example mini batch info for full MlodModel training.
        This includes ground truth, ortho rotated ground truth,
        negative proposal anchors, positive proposal anchors, and a sampled
        mini batch.

        The 2D iou can be modified to show the effect of changing the iou
        threshold for mini batch sampling.

        In order to let this demo run without training an RPN, the proposals
        shown are being read from a text file.

    Keys:
        F1: Toggle ground truth
        F2: Toggle ortho rotated ground truth
        F3: Toggle negative proposal anchors
        F4: Toggle positive proposal anchors
        F5: Toggle mini batch anchors
    """

    ##############################
    #  Options
    ##############################
    # Config file folder, default (<mlod_root>/data/outputs/<checkpoint_name>)
    config_dir = None

    # checkpoint_name = None
    checkpoint_name = 'mlod_exp_example'
    data_split = 'val_half'

    # global_step = None
    global_step = 100000

    # # # Cars # # #
    # sample_name = "000050"
    sample_name = "000104"
    # sample_name = "000764"

    # # # People # # #
    # val_half
    # sample_name = '000001'  # Hard, 1 far cyc
    # sample_name = '000005'  # Easy, 1 ped
    # sample_name = '000122'  # Easy, 1 cyc
    # sample_name = '000134'  # Hard, lots of people
    # sample_name = '000167'  # Medium, 1 ped, 2 cycs
    # sample_name = '000187'  # Medium, 1 ped on left
    # sample_name = '000381'  # Easy, 1 ped
    # sample_name = '000398'  # Easy, 1 ped
    # sample_name = '000401'  # Hard, obscured peds
    # sample_name = '000407'  # Easy, 1 ped
    # sample_name = '000448'  # Hard, several far people
    # sample_name = '000486'  # Hard 2 obscured peds
    # sample_name = '000509'  # Easy, 1 ped
    # sample_name = '000718'  # Hard, lots of people
    # sample_name = '002216'  # Easy, 1 cyc

    mini_batch_size = 512
    neg_proposal_2d_iou_hi = 0.6
    pos_proposal_2d_iou_lo = 0.65

    bkg_proposals_line_width = 0.5
    neg_proposals_line_width = 0.5
    mid_proposals_line_width = 0.5
    pos_proposals_line_width = 1.0

    ##############################
    # End of Options
    ##############################

    img_idx = int(sample_name)
    print("Showing mini batch for sample {}".format(sample_name))

    # Read proposals from file
    if checkpoint_name is None:
        # Use VAL Dataset
        dataset = DatasetBuilder.build_kitti_dataset(
            DatasetBuilder.KITTI_VAL)

        # Load demo proposals
        proposals_and_scores_dir = mlod.top_dir() + \
            '/demos/data/predictions/' + checkpoint_name + \
            '/proposals_and_scores/' + dataset.data_split
    else:
        if config_dir is None:
            config_dir = mlod.root_dir() + '/data/outputs/' + checkpoint_name

        # Parse experiment config
        pipeline_config_file = \
            config_dir + '/' + checkpoint_name + '.config'
        _, _, _, dataset_config = \
            config_builder_util.get_configs_from_pipeline_file(
                pipeline_config_file, is_training=False)

        dataset_config.data_split = data_split
        dataset = DatasetBuilder.build_kitti_dataset(dataset_config,
                                                     use_defaults=False)

        # Overwrite
        mini_batch_utils = dataset.kitti_utils.mini_batch_utils
        mini_batch_utils.mlod_neg_iou_range[1] = neg_proposal_2d_iou_hi
        mini_batch_utils.mlod_pos_iou_range[0] = pos_proposal_2d_iou_lo

        # Load proposals from outputs folder
        proposals_and_scores_dir = mlod.root_dir() + \
            '/data/outputs/' + checkpoint_name + \
            '/predictions/proposals_and_scores/' + dataset.data_split

    # Get checkpoint step
    steps = os.listdir(proposals_and_scores_dir)
    steps.sort(key=int)
    print('Available steps: {}'.format(steps))

    # Use latest checkpoint if no index provided
    if global_step is None:
        global_step = steps[-1]

    proposals_and_scores = np.loadtxt(proposals_and_scores_dir +
                                      "/{}/{}.txt".format(global_step,
                                                          sample_name))
    proposal_boxes_3d = proposals_and_scores[:, 0:7]
    proposal_anchors = box_3d_encoder.box_3d_to_anchor(proposal_boxes_3d)

    # Get filtered ground truth
    obj_labels = obj_utils.read_labels(dataset.label_dir, img_idx)
    filtered_objs = dataset.kitti_utils.filter_labels(obj_labels)

    # Convert ground truth to anchors
    gt_boxes_3d = np.asarray(
        [box_3d_encoder.object_label_to_box_3d(obj_label)
         for obj_label in filtered_objs])
    gt_anchors = box_3d_encoder.box_3d_to_anchor(gt_boxes_3d,
                                                 ortho_rotate=True)

    # Ortho rotate ground truth
    gt_ortho_boxes_3d = box_3d_encoder.anchors_to_box_3d(gt_anchors)
    gt_ortho_objs = [box_3d_encoder.box_3d_to_object_label(box_3d,
                                                           obj_type='OrthoGt')
                     for box_3d in gt_ortho_boxes_3d]

    # Project gt and anchors into BEV
    gt_bev_anchors, _ = \
        anchor_projector.project_to_bev(gt_anchors,
                                        dataset.kitti_utils.bev_extents)
    bev_anchors, _ = \
        anchor_projector.project_to_bev(proposal_anchors,
                                        dataset.kitti_utils.bev_extents)

    # Reorder boxes into (y1, x1, y2, x2) order
    gt_bev_anchors_tf_order = anchor_projector.reorder_projected_boxes(
        gt_bev_anchors)
    bev_anchors_tf_order = anchor_projector.reorder_projected_boxes(
        bev_anchors)

    # Convert to box_list format for iou calculation
    gt_anchor_box_list = box_list.BoxList(tf.cast(gt_bev_anchors_tf_order,
                                                  tf.float32))
    anchor_box_list = box_list.BoxList(tf.cast(bev_anchors_tf_order,
                                               tf.float32))

    # Get IoU for every anchor
    tf_all_ious = box_list_ops.iou(gt_anchor_box_list, anchor_box_list)
    valid_ious = True
    # Make sure the calculated IoUs contain values. Since its a [N, M]
    # tensor, if there are no gt's for instance, that entry will be zero.
    if tf_all_ious.shape[0] == 0 or tf_all_ious.shape[1] == 0:
        print('#################################################')
        print('Warning: This sample does not contain valid IoUs')
        print('#################################################')
        valid_ious = False

    if valid_ious:
        tf_max_ious = tf.reduce_max(tf_all_ious, axis=0)
        tf_max_iou_indices = tf.argmax(tf_all_ious, axis=0)

        # Sample an RPN mini batch from the non empty anchors
        mini_batch_utils = dataset.kitti_utils.mini_batch_utils

        # Overwrite mini batch size and sample a mini batch
        mini_batch_utils.mlod_mini_batch_size = mini_batch_size
        mb_mask_tf, _ = mini_batch_utils.sample_mlod_mini_batch(tf_max_ious)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # Run the graph to calculate ious for every proposal and
        # to get the mini batch mask
        all_ious, max_ious, max_iou_indices = sess.run([tf_all_ious,
                                                        tf_max_ious,
                                                        tf_max_iou_indices])
        mb_mask = sess.run(mb_mask_tf)

        mb_anchors = proposal_anchors[mb_mask]
        mb_anchor_boxes_3d = box_3d_encoder.anchors_to_box_3d(mb_anchors)
        mb_anchor_ious = max_ious[mb_mask]

    else:
        # We have no valid IoU's, so assume all IoUs are zeros
        # and the mini-batch contains all the anchors since we cannot
        # mask without IoUs.
        max_ious = np.zeros(proposal_boxes_3d.shape[0])
        mb_anchor_ious = max_ious
        mb_anchors = proposal_anchors
        mb_anchor_boxes_3d = box_3d_encoder.anchors_to_box_3d(mb_anchors)

    # Create list of positive/negative proposals based on iou
    pos_proposal_objs = []
    mid_proposal_objs = []
    neg_proposal_objs = []
    bkg_proposal_objs = []
    for i in range(len(proposal_boxes_3d)):
        box_3d = proposal_boxes_3d[i]

        if max_ious[i] == 0.0:
            # Background proposals
            bkg_proposal_objs.append(
                box_3d_encoder.box_3d_to_object_label(
                    box_3d,
                    obj_type='BackgroundProposal'))

        elif max_ious[i] < neg_proposal_2d_iou_hi:
            # Negative proposals
            neg_proposal_objs.append(
                box_3d_encoder.box_3d_to_object_label(
                    box_3d,
                    obj_type='NegativeProposal'))

        elif max_ious[i] < pos_proposal_2d_iou_lo:
            # Middle proposals (in between negative and positive)
            mid_proposal_objs.append(
                box_3d_encoder.box_3d_to_object_label(
                    box_3d,
                    obj_type='MiddleProposal'))

        elif max_ious[i] <= 1.0:
            # Positive proposals
            pos_proposal_objs.append(
                box_3d_encoder.box_3d_to_object_label(
                    box_3d,
                    obj_type='PositiveProposal'))

        else:
            raise ValueError('Invalid IoU > 1.0')

    print('{} bkg, {} neg, {} mid, {} pos proposals:'.format(
        len(bkg_proposal_objs), len(neg_proposal_objs),
        len(mid_proposal_objs), len(pos_proposal_objs)))

    # Convert the mini_batch anchors to object list
    mb_obj_list = []
    for i in range(len(mb_anchor_ious)):
        if valid_ious and (mb_anchor_ious[i] >
                           mini_batch_utils.mlod_pos_iou_range[0]):
            obj_type = "Positive"
        else:
            obj_type = "Negative"

        obj = box_3d_encoder.box_3d_to_object_label(mb_anchor_boxes_3d[i],
                                                    obj_type)
        mb_obj_list.append(obj)

    # Point cloud
    image = cv2.imread(dataset.get_rgb_image_path(sample_name))
    points, point_colours = demo_utils.get_filtered_pc_and_colours(dataset,
                                                                   image,
                                                                   img_idx)

    # Visualize from here
    vis_utils.visualization(dataset.rgb_image_dir, img_idx)
    plt.show(block=False)

    # VtkPointCloud
    vtk_point_cloud = VtkPointCloud()
    vtk_point_cloud.set_points(points, point_colours)

    # VtkAxes
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(5, 5, 5)

    # VtkBoxes for ground truth
    vtk_gt_boxes = VtkBoxes()
    vtk_gt_boxes.set_objects(filtered_objs, COLOUR_SCHEME)

    # VtkBoxes for ortho ground truth
    vtk_gt_ortho_boxes = VtkBoxes()
    vtk_gt_ortho_boxes.set_objects(gt_ortho_objs, COLOUR_SCHEME)

    # VtkBoxes for background proposals
    vtk_bkg_proposal_boxes = VtkBoxes()
    vtk_bkg_proposal_boxes.set_objects(bkg_proposal_objs, COLOUR_SCHEME)
    vtk_bkg_proposal_boxes.set_line_width(bkg_proposals_line_width)

    # VtkBoxes for negative proposals
    vtk_neg_proposal_boxes = VtkBoxes()
    vtk_neg_proposal_boxes.set_objects(neg_proposal_objs, COLOUR_SCHEME)
    vtk_neg_proposal_boxes.set_line_width(neg_proposals_line_width)

    # VtkBoxes for middle proposals
    vtk_mid_proposal_boxes = VtkBoxes()
    vtk_mid_proposal_boxes.set_objects(mid_proposal_objs, COLOUR_SCHEME)
    vtk_mid_proposal_boxes.set_line_width(mid_proposals_line_width)

    # VtkBoxes for positive proposals
    vtk_pos_proposal_boxes = VtkBoxes()
    vtk_pos_proposal_boxes.set_objects(pos_proposal_objs, COLOUR_SCHEME)
    vtk_pos_proposal_boxes.set_line_width(pos_proposals_line_width)

    # Create VtkBoxes for mini batch anchors
    vtk_mb_boxes = VtkBoxes()
    vtk_mb_boxes.set_objects(mb_obj_list, COLOUR_SCHEME)

    # Create Voxel Grid Renderer in bottom half
    vtk_renderer = vtk.vtkRenderer()
    vtk_renderer.SetBackground(0.2, 0.3, 0.4)

    # Add actors
    vtk_renderer.AddActor(axes)
    vtk_renderer.AddActor(vtk_point_cloud.vtk_actor)

    vtk_renderer.AddActor(vtk_gt_boxes.vtk_actor)
    vtk_renderer.AddActor(vtk_gt_ortho_boxes.vtk_actor)

    vtk_renderer.AddActor(vtk_bkg_proposal_boxes.vtk_actor)
    vtk_renderer.AddActor(vtk_neg_proposal_boxes.vtk_actor)
    vtk_renderer.AddActor(vtk_mid_proposal_boxes.vtk_actor)
    vtk_renderer.AddActor(vtk_pos_proposal_boxes.vtk_actor)

    vtk_renderer.AddActor(vtk_mb_boxes.vtk_actor)

    # Setup Camera
    current_cam = vtk_renderer.GetActiveCamera()
    current_cam.Pitch(160.0)
    current_cam.Roll(180.0)

    # Zooms out to fit all points on screen
    vtk_renderer.ResetCamera()

    # Zoom in slightly
    current_cam.Zoom(2.5)

    # Reset the clipping range to show all points
    vtk_renderer.ResetCameraClippingRange()

    # Setup Render Window
    vtk_render_window = vtk.vtkRenderWindow()
    vtk_render_window.SetWindowName("MLOD Mini Batch")
    vtk_render_window.SetSize(900, 500)
    vtk_render_window.AddRenderer(vtk_renderer)

    # Setup custom interactor style, which handles mouse and key events
    vtk_render_window_interactor = vtk.vtkRenderWindowInteractor()
    vtk_render_window_interactor.SetRenderWindow(vtk_render_window)

    vtk_render_window_interactor.SetInteractorStyle(
        vis_utils.ToggleActorsInteractorStyle([
            vtk_gt_boxes.vtk_actor,
            vtk_gt_ortho_boxes.vtk_actor,

            vtk_bkg_proposal_boxes.vtk_actor,
            vtk_neg_proposal_boxes.vtk_actor,
            vtk_mid_proposal_boxes.vtk_actor,
            vtk_pos_proposal_boxes.vtk_actor,

            vtk_mb_boxes.vtk_actor,
        ]))

    # Render in VTK
    vtk_render_window.Render()
    vtk_render_window_interactor.Start()


if __name__ == '__main__':
    main()
