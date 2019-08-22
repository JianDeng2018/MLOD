import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import vtk
from wavedata.tools.core import calib_utils

from wavedata.tools.obj_detection import obj_utils
from wavedata.tools.visualization import vis_utils
from wavedata.tools.visualization.vtk_boxes import VtkBoxes
from wavedata.tools.visualization.vtk_ground_plane import VtkGroundPlane
from wavedata.tools.visualization.vtk_point_cloud import VtkPointCloud

import mlod
from mlod.builders.dataset_builder import DatasetBuilder
from mlod.core import anchor_encoder
from mlod.core import box_3d_encoder
from mlod.core.anchor_generators import grid_anchor_3d_generator
from mlod.datasets.kitti import kitti_aug


def main():
    """Flip RPN Mini Batch
     Visualization of the mini batch anchors for RpnModel training.

     Keys:
         F1: Toggle mini batch anchors
         F2: Flipped
     """

    anchor_colour_scheme = {
        "Car": (255, 0, 0),  # Red
        "Pedestrian": (255, 150, 50),  # Orange
        "Cyclist": (150, 50, 100),  # Purple
        "DontCare": (255, 255, 255),  # White

        "Anchor": (150, 150, 150),  # Gray
        "Regressed Anchor": (255, 255, 0),  # Yellow

        "Positive": (0, 255, 255),  # Teal
        "Negative": (255, 0, 255)  # Purple
    }

    dataset_config_path = mlod.root_dir() + \
        '/configs/mb_rpn_demo_cars.config'

    # dataset_config_path = mlod.root_dir() + \
    #     '/configs/mb_rpn_demo_people.config'

    ##############################
    # Options
    ##############################
    # # # Random sample # # #
    sample_name = None

    # # # Cars # # #
    # sample_name = "000001"
    # sample_name = "000050"
    # sample_name = "000104"
    # sample_name = "000112"
    # sample_name = "000169"
    # sample_name = "000191"

    sample_name = "003801"

    # # # Pedestrians # # #
    # sample_name = "000000"
    # sample_name = "000011"
    # sample_name = "000015"
    # sample_name = "000028"
    # sample_name = "000035"
    # sample_name = "000134"
    # sample_name = "000167"
    # sample_name = '000379'
    # sample_name = '000381'
    # sample_name = '000397'
    # sample_name = '000398'
    # sample_name = '000401'
    # sample_name = '000407'
    # sample_name = '000486'
    # sample_name = '000509'

    # # Cyclists # # #
    # sample_name = '000122'
    # sample_name = '000448'

    # # # Multiple classes # # #
    # sample_name = "000764"
    ##############################
    # End of Options
    ##############################

    # Create Dataset
    dataset = DatasetBuilder.load_dataset_from_config(dataset_config_path)

    # Random sample
    if sample_name is None:
        sample_idx = np.random.randint(0, dataset.num_samples)
        sample_name = dataset.sample_list[sample_idx]

    anchor_strides = dataset.kitti_utils.anchor_strides

    img_idx = int(sample_name)

    print("Showing mini batch for sample {}".format(sample_name))

    image = cv2.imread(dataset.get_rgb_image_path(sample_name))
    image_shape = [image.shape[1], image.shape[0]]

    # KittiUtils class
    dataset_utils = dataset.kitti_utils

    ground_plane = obj_utils.get_road_plane(img_idx, dataset.planes_dir)

    point_cloud = obj_utils.get_depth_map_point_cloud(img_idx,
                                                      dataset.calib_dir,
                                                      dataset.depth_dir,
                                                      image_shape)
    points = point_cloud.T

    # Grab ground truth
    ground_truth_list = obj_utils.read_labels(dataset.label_dir, img_idx)
    ground_truth_list = dataset_utils.filter_labels(ground_truth_list)

    stereo_calib_p2 = calib_utils.read_calibration(dataset.calib_dir,
                                                   img_idx).p2

    ##############################
    # Flip sample info
    ##############################
    start_time = time.time()

    flipped_image = kitti_aug.flip_image(image)
    flipped_point_cloud = kitti_aug.flip_point_cloud(point_cloud)
    flipped_gt_list = [kitti_aug.flip_label_in_3d_only(obj)
                       for obj in ground_truth_list]
    flipped_ground_plane = kitti_aug.flip_ground_plane(ground_plane)
    flipped_calib_p2 = kitti_aug.flip_stereo_calib_p2(
        stereo_calib_p2, image_shape)

    print('flip sample', time.time() - start_time)

    flipped_points = flipped_point_cloud.T
    point_colours = vis_utils.project_img_to_point_cloud(points,
                                                         image,
                                                         dataset.calib_dir,
                                                         img_idx)

    ##############################
    # Generate anchors
    ##############################
    clusters, _ = dataset.get_cluster_info()
    anchor_generator = grid_anchor_3d_generator.GridAnchor3dGenerator()

    # Read mini batch info
    anchors_info = dataset_utils.get_anchors_info(sample_name)

    all_anchor_boxes_3d = []
    all_ious = []
    all_offsets = []
    for class_idx in range(len(dataset.classes)):

        anchor_boxes_3d = anchor_generator.generate(
            area_3d=dataset.kitti_utils.area_extents,
            anchor_3d_sizes=clusters[class_idx],
            anchor_stride=anchor_strides[class_idx],
            ground_plane=ground_plane)

        if len(anchors_info[class_idx]) > 0:
            indices, ious, offsets, classes = anchors_info[class_idx]

            # Get non empty anchors from the indices
            non_empty_anchor_boxes_3d = anchor_boxes_3d[indices]

            all_anchor_boxes_3d.extend(non_empty_anchor_boxes_3d)
            all_ious.extend(ious)
            all_offsets.extend(offsets)

    if not len(all_anchor_boxes_3d) > 0:
        # Exit early if anchors_info is empty
        print("No anchors, Please try a different sample")
        return

    # Convert to ndarrays
    all_anchor_boxes_3d = np.asarray(all_anchor_boxes_3d)
    all_ious = np.asarray(all_ious)
    all_offsets = np.asarray(all_offsets)

    ##############################
    # Flip anchors
    ##############################
    start_time = time.time()

    # Flip anchors and offsets
    flipped_anchor_boxes_3d = kitti_aug.flip_boxes_3d(all_anchor_boxes_3d,
                                                      flip_ry=False)
    all_offsets[:, 0] = -all_offsets[:, 0]

    print('flip anchors and offsets', time.time() - start_time)

    # Overwrite with flipped things
    all_anchor_boxes_3d = flipped_anchor_boxes_3d
    points = flipped_points
    ground_truth_list = flipped_gt_list
    ground_plane = flipped_ground_plane

    ##############################
    # Mini batch sampling
    ##############################
    # Sample an RPN mini batch from the non empty anchors
    mini_batch_utils = dataset.kitti_utils.mini_batch_utils
    mb_mask_tf, _ = mini_batch_utils.sample_rpn_mini_batch(all_ious)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    mb_mask = sess.run(mb_mask_tf)

    mb_anchor_boxes_3d = all_anchor_boxes_3d[mb_mask]
    mb_anchor_ious = all_ious[mb_mask]
    mb_anchor_offsets = all_offsets[mb_mask]

    # ObjectLabel list that hold all boxes to visualize
    obj_list = []

    # Convert the mini_batch anchors to object list
    for i in range(len(mb_anchor_boxes_3d)):
        if mb_anchor_ious[i] > mini_batch_utils.rpn_pos_iou_range[0]:
            obj_type = "Positive"
        else:
            obj_type = "Negative"

        obj = box_3d_encoder.box_3d_to_object_label(mb_anchor_boxes_3d[i],
                                                    obj_type)
        obj_list.append(obj)

    # Convert all non-empty anchors to object list
    non_empty_anchor_objs = \
        [box_3d_encoder.box_3d_to_object_label(
            anchor_box_3d, obj_type='Anchor')
         for anchor_box_3d in all_anchor_boxes_3d]

    ##############################
    # Regress Positive Anchors
    ##############################
    # Convert anchor_boxes_3d to anchors and apply offsets
    mb_pos_mask = mb_anchor_ious > mini_batch_utils.rpn_pos_iou_range[0]
    mb_pos_anchor_boxes_3d = mb_anchor_boxes_3d[mb_pos_mask]
    mb_pos_anchor_offsets = mb_anchor_offsets[mb_pos_mask]

    mb_pos_anchors = box_3d_encoder.box_3d_to_anchor(mb_pos_anchor_boxes_3d)
    regressed_pos_anchors = anchor_encoder.offset_to_anchor(
        mb_pos_anchors, mb_pos_anchor_offsets)

    # Convert regressed anchors to ObjectLabels for visualization
    regressed_anchor_boxes_3d = box_3d_encoder.anchors_to_box_3d(
        regressed_pos_anchors, fix_lw=True)
    regressed_anchor_objs = \
        [box_3d_encoder.box_3d_to_object_label(
            box_3d, obj_type='Regressed Anchor')
         for box_3d in regressed_anchor_boxes_3d]

    ##############################
    # Visualization
    ##############################
    cv2.imshow('{} flipped'.format(sample_name), flipped_image)
    cv2.waitKey()

    # Create VtkAxes
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(5, 5, 5)

    # Create VtkBoxes for mini batch anchors
    vtk_pos_anchor_boxes = VtkBoxes()
    vtk_pos_anchor_boxes.set_objects(obj_list, anchor_colour_scheme)

    # VtkBoxes for non empty anchors
    vtk_non_empty_anchors = VtkBoxes()
    vtk_non_empty_anchors.set_objects(non_empty_anchor_objs,
                                      anchor_colour_scheme)
    vtk_non_empty_anchors.set_line_width(0.1)

    # VtkBoxes for regressed anchors
    vtk_regressed_anchors = VtkBoxes()
    vtk_regressed_anchors.set_objects(regressed_anchor_objs,
                                      anchor_colour_scheme)
    vtk_regressed_anchors.set_line_width(5.0)

    # Create VtkBoxes for ground truth
    vtk_gt_boxes = VtkBoxes()
    vtk_gt_boxes.set_objects(ground_truth_list, anchor_colour_scheme,
                             show_orientations=True)

    vtk_point_cloud = VtkPointCloud()
    vtk_point_cloud.set_points(points, point_colours)

    vtk_ground_plane = VtkGroundPlane()
    vtk_ground_plane.set_plane(ground_plane, dataset.kitti_utils.bev_extents)

    # Create Voxel Grid Renderer in bottom half
    vtk_renderer = vtk.vtkRenderer()

    vtk_renderer.AddActor(vtk_point_cloud.vtk_actor)
    vtk_renderer.AddActor(vtk_non_empty_anchors.vtk_actor)
    vtk_renderer.AddActor(vtk_pos_anchor_boxes.vtk_actor)
    vtk_renderer.AddActor(vtk_regressed_anchors.vtk_actor)
    vtk_renderer.AddActor(vtk_gt_boxes.vtk_actor)
    vtk_renderer.AddActor(vtk_ground_plane.vtk_actor)

    vtk_renderer.AddActor(axes)
    vtk_renderer.SetBackground(0.2, 0.3, 0.4)

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
    vtk_render_window.SetWindowName("RPN Mini Batch")
    vtk_render_window.SetSize(900, 500)
    vtk_render_window.AddRenderer(vtk_renderer)

    # Setup custom interactor style, which handles mouse and key events
    vtk_render_window_interactor = vtk.vtkRenderWindowInteractor()
    vtk_render_window_interactor.SetRenderWindow(vtk_render_window)

    vtk_render_window_interactor.SetInteractorStyle(
        vis_utils.ToggleActorsInteractorStyle([
            vtk_non_empty_anchors.vtk_actor,
            vtk_pos_anchor_boxes.vtk_actor,
            vtk_regressed_anchors.vtk_actor,
            vtk_ground_plane.vtk_actor,
        ]))

    # Render in VTK
    vtk_render_window.Render()
    vtk_render_window_interactor.Start()


if __name__ == '__main__':
    main()
