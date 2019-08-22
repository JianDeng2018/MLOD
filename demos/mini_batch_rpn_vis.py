import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import vtk

from wavedata.tools.obj_detection import obj_utils
from wavedata.tools.visualization import vis_utils
from wavedata.tools.visualization.vtk_boxes import VtkBoxes
from wavedata.tools.visualization.vtk_ground_plane import VtkGroundPlane
from wavedata.tools.visualization.vtk_point_cloud import VtkPointCloud

import mlod
from mlod.builders.dataset_builder import DatasetBuilder
from mlod.core import box_3d_encoder
from mlod.core.anchor_generators import grid_anchor_3d_generator
from mlod.utils import demo_utils


def main():
    """
     Visualization of the mini batch anchors for RpnModel training.

     Keys:
        F1: Toggle mini batch anchors
        F2: Toggle positive/negative proposal anchors
        F3: Toggle easy ground truth objects (Green)
        F4: Toggle medium ground truth objects (Orange)
        F5: Toggle hard ground truth objects (Red)
        F6: Toggle all ground truth objects (default off)
        F7: Toggle ground-plane
     """

    anchor_colour_scheme = {
        "Car": (255, 0, 0),             # Red
        "Pedestrian": (255, 150, 50),   # Orange
        "Cyclist": (150, 50, 100),      # Purple
        "DontCare": (255, 255, 255),    # White

        "Anchor": (150, 150, 150),      # Gray

        "Positive": (0, 255, 255),      # Teal
        "Negative": (255, 0, 255)       # Bright Purple
    }

    ##############################
    # Options
    ##############################
    show_orientations = True

    # Classes name
    config_name = 'car'
    # config_name = 'ped'
    # config_name = 'cyc'
    # config_name = 'ppl'

    # # # Random sample # # #
    sample_name = None

    # Small cars
    # sample_name = '000008'
    # sample_name = '000639'

    # # # Cars # # #
    # sample_name = "000001"
    # sample_name = "000050"
    # sample_name = "000112"
    # sample_name = "000169"
    # sample_name = "000191"

    # # # People # # #
    # sample_name = '000000'

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
    sample_name = '000448'  # Hard, several far people
    # sample_name = '000486'  # Hard 2 obscured peds
    # sample_name = '000509'  # Easy, 1 ped
    # sample_name = '000718'  # Hard, lots of people
    # sample_name = '002216'  # Easy, 1 cyc

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

    # Dataset config
    dataset_config_path = mlod.top_dir() + \
        '/demos/configs/mb_rpn_{}.config'.format(config_name)

    # Create Dataset
    dataset = DatasetBuilder.load_dataset_from_config(
        dataset_config_path)

    # Random sample
    if sample_name is None:
        sample_idx = np.random.randint(0, dataset.num_samples)
        sample_name = dataset.sample_list[sample_idx].name

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
    point_colours = vis_utils.project_img_to_point_cloud(points, image,
                                                         dataset.calib_dir,
                                                         img_idx)

    clusters, _ = dataset.get_cluster_info()
    anchor_generator = grid_anchor_3d_generator.GridAnchor3dGenerator()

    # Read mini batch info
    anchors_info = dataset_utils.get_anchors_info(
        dataset.classes_name, anchor_strides, sample_name)

    if not anchors_info:
        # Exit early if anchors_info is empty
        print("Anchors info is empty, please try a different sample")
        return

    # Generate anchors for all classes
    all_anchor_boxes_3d = []
    for class_idx in range(len(dataset.classes)):

        anchor_boxes_3d = anchor_generator.generate(
            area_3d=dataset.kitti_utils.area_extents,
            anchor_3d_sizes=clusters[class_idx],
            anchor_stride=anchor_strides[class_idx],
            ground_plane=ground_plane)

        all_anchor_boxes_3d.extend(anchor_boxes_3d)
    all_anchor_boxes_3d = np.asarray(all_anchor_boxes_3d)

    # Use anchors info
    indices, ious, offsets, classes = anchors_info

    # Get non empty anchors from the indices
    anchor_boxes_3d = all_anchor_boxes_3d[indices]

    # Sample an RPN mini batch from the non empty anchors
    mini_batch_utils = dataset.kitti_utils.mini_batch_utils
    mb_mask_tf, _ = mini_batch_utils.sample_rpn_mini_batch(ious)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    mb_mask = sess.run(mb_mask_tf)

    mb_anchor_boxes_3d = anchor_boxes_3d[mb_mask]
    mb_anchor_ious = ious[mb_mask]

    # ObjectLabel list that hold all boxes to visualize
    obj_list = []

    num_positives = 0
    # Convert the mini_batch anchors to object list
    mini_batch_size = mini_batch_utils.rpn_mini_batch_size
    for i in range(mini_batch_size):
        if mb_anchor_ious[i] > mini_batch_utils.rpn_pos_iou_range[0]:
            obj_type = "Positive"
            num_positives += 1
        else:
            obj_type = "Negative"

        obj = box_3d_encoder.box_3d_to_object_label(mb_anchor_boxes_3d[i],
                                                    obj_type)
        obj_list.append(obj)

    print('Num positives', num_positives)

    # Convert all non-empty anchors to object list
    non_empty_anchor_objs = \
        [box_3d_encoder.box_3d_to_object_label(
            anchor_box_3d, obj_type='Anchor')
         for anchor_box_3d in anchor_boxes_3d]

    ##############################
    # Ground Truth
    ##############################
    if dataset.has_labels:
        easy_gt_objs, medium_gt_objs, \
            hard_gt_objs, all_gt_objs = demo_utils.get_gts_based_on_difficulty(
                dataset, img_idx)
    else:
        easy_gt_objs = medium_gt_objs = hard_gt_objs = all_gt_objs = []

    # Visualize 2D image
    vis_utils.visualization(dataset.rgb_image_dir, img_idx)
    plt.show(block=False)

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

    # Create VtkBoxes for ground truth
    vtk_easy_gt_boxes, vtk_medium_gt_boxes, \
        vtk_hard_gt_boxes, vtk_all_gt_boxes = \
        demo_utils.create_gt_vtk_boxes(easy_gt_objs,
                                       medium_gt_objs,
                                       hard_gt_objs,
                                       all_gt_objs,
                                       show_orientations)

    vtk_point_cloud = VtkPointCloud()
    vtk_point_cloud.set_points(points, point_colours)
    vtk_point_cloud.vtk_actor.GetProperty().SetPointSize(2)

    vtk_ground_plane = VtkGroundPlane()
    vtk_ground_plane.set_plane(ground_plane, dataset.kitti_utils.bev_extents)

    # vtk_voxel_grid = VtkVoxelGrid()
    # vtk_voxel_grid.set_voxels(vx_grid)

    # Create Voxel Grid Renderer in bottom half
    vtk_renderer = vtk.vtkRenderer()
    vtk_renderer.AddActor(vtk_point_cloud.vtk_actor)
    vtk_renderer.AddActor(vtk_ground_plane.vtk_actor)

    vtk_renderer.AddActor(vtk_hard_gt_boxes.vtk_actor)
    vtk_renderer.AddActor(vtk_medium_gt_boxes.vtk_actor)
    vtk_renderer.AddActor(vtk_easy_gt_boxes.vtk_actor)
    vtk_renderer.AddActor(vtk_all_gt_boxes.vtk_actor)

    # vtk_renderer.AddActor(vtk_voxel_grid.vtk_actor)
    vtk_renderer.AddActor(vtk_non_empty_anchors.vtk_actor)
    vtk_renderer.AddActor(vtk_pos_anchor_boxes.vtk_actor)
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
    mb_iou_thresholds = np.round(
        [mini_batch_utils.rpn_neg_iou_range[1],
         mini_batch_utils.rpn_pos_iou_range[0]], 3)
    vtk_render_window.SetWindowName(
        'Sample {} RPN Mini Batch {}/{}, '
        'Num Positives {}'.format(
            sample_name,
            mb_iou_thresholds[0],
            mb_iou_thresholds[1],
            num_positives))
    vtk_render_window.SetSize(900, 500)
    vtk_render_window.AddRenderer(vtk_renderer)

    # Setup custom interactor style, which handles mouse and key events
    vtk_render_window_interactor = vtk.vtkRenderWindowInteractor()
    vtk_render_window_interactor.SetRenderWindow(vtk_render_window)

    vtk_render_window_interactor.SetInteractorStyle(
        vis_utils.ToggleActorsInteractorStyle([
            vtk_non_empty_anchors.vtk_actor,
            vtk_pos_anchor_boxes.vtk_actor,

            vtk_easy_gt_boxes.vtk_actor,
            vtk_medium_gt_boxes.vtk_actor,
            vtk_hard_gt_boxes.vtk_actor,
            vtk_all_gt_boxes.vtk_actor,

            vtk_ground_plane.vtk_actor
        ]))

    # Render in VTK
    vtk_render_window.Render()
    vtk_render_window_interactor.Start()


if __name__ == '__main__':
    main()
