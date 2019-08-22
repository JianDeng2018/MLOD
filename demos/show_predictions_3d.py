import copy
import cv2
import numpy as np
import os
import vtk

from wavedata.tools.obj_detection import evaluation
from wavedata.tools.core.voxel_grid import VoxelGrid
from wavedata.tools.obj_detection import obj_utils

from wavedata.tools.visualization import vis_utils
from wavedata.tools.visualization.vtk_boxes import VtkBoxes
from wavedata.tools.visualization.vtk_pyramid_boxes import VtkPyramidBoxes
from wavedata.tools.visualization.vtk_ground_plane import VtkGroundPlane
from wavedata.tools.visualization.vtk_point_cloud import VtkPointCloud
from wavedata.tools.visualization.vtk_text_labels import VtkTextLabels
from wavedata.tools.visualization.vtk_voxel_grid import VtkVoxelGrid

import mlod
from mlod.builders import config_builder_util
from mlod.builders.dataset_builder import DatasetBuilder
from mlod.core import box_3d_encoder
from mlod.utils import demo_utils

COLOUR_SCHEME_PREDICTIONS = {
    "Easy GT": (255, 255, 0),     # Yellow
    "Medium GT": (255, 128, 0),   # Orange
    "Hard GT": (255, 0, 0),       # Red

    "Proposal": (255, 0, 127),    # Pink

    "Car": (50, 255, 50),         # Green
    "Pedestrian": (0, 255, 255),  # Teal
    "Cyclist": (255, 255, 0),     # Yellow
}


def main():
    """This demo shows RPN proposals and MLOD predictions in the
    3D point cloud.

    Keys:
        F1: Toggle proposals
        F2: Toggle predictions
        F3: Toggle 3D voxel grid
        F4: Toggle point cloud

        F5: Toggle easy ground truth objects (Green)
        F6: Toggle medium ground truth objects (Orange)
        F7: Toggle hard ground truth objects (Red)
        F8: Toggle all ground truth objects (default off)

        F9: Toggle ground slice filter (default off)
        F10: Toggle offset slice filter (default off)
    """

    ##############################
    # Options
    ##############################
    rpn_score_threshold = 0.1
    mlod_score_threshold = 0.1

    proposals_line_width = 1.0
    predictions_line_width = 3.0
    show_orientations = True

    point_cloud_source = 'lidar'

    # Config file folder, default (<mlod_root>/data/outputs/<checkpoint_name>)
    config_dir = None

    checkpoint_name = 'mlod_fpn_people_n_m'
    global_step = 135000  # Latest checkpoint

    #data_split = 'val_half'
    #data_split = 'val'
    data_split = 'test'

    # Show 3D iou text
    draw_ious_3d = False

    sample_name = '000031'

    # # # Cars # # #
    # sample_name = '000050'
    # sample_name = '000104'
    # sample_name = '000169'
    # sample_name = '000191'
    # sample_name = '000360'
    # sample_name = '001783'
    # sample_name = '001820'

    # val split
    # sample_name = '000181'
    # sample_name = '000751'
    # sample_name = '000843'
    # sample_name = '000944'
    # sample_name = '006338'

    # # # People # # #
    # val_half split
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

    # val split
    # sample_name = '000015'
    # sample_name = '000048'
    # sample_name = '000058'
    # sample_name = '000076'    # Medium, few ped, 1 cyc
    # sample_name = '000108'
    # sample_name = '000118'
    # sample_name = '000145'
    # sample_name = '000153'
    # sample_name = '000186'
    # sample_name = '000195'
    # sample_name = '000199'
    # sample_name = '000397'
    # sample_name = '004425'
    # sample_name = '004474'    # Hard, many ped, 1 cyc
    # sample_name = '004657'    # Hard, Few cycl, few ped
    # sample_name = '006071'
    # sample_name = '006828'    # Hard, Few cycl, few ped
    # sample_name = '006908'    # Hard, Few cycl, few ped
    # sample_name = '007412'
    # sample_name = '007318'    # Hard, Few cycl, few ped

    ##############################
    # End of Options
    ##############################

    if data_split == 'test':
        draw_ious_3d = False

    if config_dir is None:
        config_dir = mlod.root_dir() + '/data/outputs/' + checkpoint_name

    # Parse experiment config
    pipeline_config_file = \
        config_dir + '/' + checkpoint_name + '.config'
    _, _, _, dataset_config = \
        config_builder_util.get_configs_from_pipeline_file(
            pipeline_config_file, is_training=False)

    dataset_config.data_split = data_split

    if data_split == 'test':
        dataset_config.data_split_dir = 'testing'
        dataset_config.has_labels = False

    dataset = DatasetBuilder.build_kitti_dataset(dataset_config,
                                                 use_defaults=False)



    # Random sample
    if sample_name is None:
        sample_idx = np.random.randint(0, dataset.num_samples)
        sample_name = dataset.sample_names[sample_idx]

    ##############################
    # Setup Paths
    ##############################
    img_idx = int(sample_name)

    # Text files directory
    proposals_and_scores_dir = mlod.root_dir() + \
        '/data/outputs/' + checkpoint_name + '/predictions' +  \
        '/proposals_and_scores/' + dataset.data_split

    predictions_and_scores_dir = mlod.root_dir() + \
        '/data/outputs/' + checkpoint_name + '/predictions' +  \
        '/final_predictions_and_scores/' + dataset.data_split

    # Get checkpoint step
    steps = os.listdir(proposals_and_scores_dir)
    steps.sort(key=int)
    print('Available steps: {}'.format(steps))

    # Use latest checkpoint if no index provided
    if global_step is None:
        global_step = steps[-1]

    # Output images directory
    img_out_dir = mlod.root_dir() + '/data/outputs/' + checkpoint_name + \
        '/predictions/images_3d/{}/{}/{}'.format(dataset.data_split,
                                                 global_step,
                                                 rpn_score_threshold)

    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)

    ##############################
    # Proposals
    ##############################
    # Load proposals from files
    proposals_and_scores = np.loadtxt(proposals_and_scores_dir +
                                      "/{}/{}.txt".format(global_step,
                                                          sample_name))

    proposals = proposals_and_scores[:, 0:7]
    proposal_scores = proposals_and_scores[:, 7]

    rpn_score_mask = proposal_scores > rpn_score_threshold

    proposals = proposals[rpn_score_mask]
    proposal_scores = proposal_scores[rpn_score_mask]
    print('Proposals:', len(proposal_scores), proposal_scores)

    proposal_objs = \
        [box_3d_encoder.box_3d_to_object_label(proposal,
                                               obj_type='Proposal')
         for proposal in proposals]

    ##############################
    # Predictions
    ##############################
    # Load proposals from files
    predictions_and_scores = np.loadtxt(predictions_and_scores_dir +
                                        "/{}/{}.txt".format(
                                            global_step,
                                            sample_name)).reshape(-1, 9)

    prediction_boxes_3d = predictions_and_scores[:, 0:7]
    prediction_scores = predictions_and_scores[:, 7]
    prediction_types = np.asarray(predictions_and_scores[:, 8], dtype=np.int32)

    mlod_score_mask = prediction_scores >= mlod_score_threshold
    prediction_boxes_3d = prediction_boxes_3d[mlod_score_mask]
    prediction_scores = prediction_scores[mlod_score_mask]
    print('Predictions: ', len(prediction_scores), prediction_scores)

    final_predictions = np.copy(prediction_boxes_3d)

    # # Swap l, w for predictions where w > l
    # swapped_indices = predictions[:, 4] > predictions[:, 3]
    # final_predictions[swapped_indices, 3] = predictions[swapped_indices, 4]
    # final_predictions[swapped_indices, 4] = predictions[swapped_indices, 3]

    prediction_objs = []
    dataset.classes = ['Pedestrian', 'Cyclist', 'Car']
    for pred_idx in range(len(final_predictions)):
        prediction_box_3d = final_predictions[pred_idx]
        prediction_type = dataset.classes[prediction_types[pred_idx]]
        prediction_obj = box_3d_encoder.box_3d_to_object_label(
            prediction_box_3d, obj_type=prediction_type)
        prediction_objs.append(prediction_obj)

    ##############################
    # Ground Truth
    ##############################
    dataset.has_labels = False
    if dataset.has_labels:
        # Get ground truth labels
        easy_gt_objs, medium_gt_objs, \
            hard_gt_objs, all_gt_objs = \
            demo_utils.get_gts_based_on_difficulty(dataset, img_idx)
    else:
        easy_gt_objs = medium_gt_objs = hard_gt_objs = all_gt_objs = []

    ##############################
    # 3D IoU
    ##############################
    if draw_ious_3d:
        # Convert to box_3d
        all_gt_boxes_3d = [box_3d_encoder.object_label_to_box_3d(gt_obj)
                           for gt_obj in all_gt_objs]
        pred_boxes_3d = [box_3d_encoder.object_label_to_box_3d(pred_obj)
                         for pred_obj in prediction_objs]
        max_ious_3d = demo_utils.get_max_ious_3d(all_gt_boxes_3d,
                                                 pred_boxes_3d)

    ##############################
    # Point Cloud
    ##############################
    image_path = dataset.get_rgb_image_path(sample_name)
    image = cv2.imread(image_path)

    point_cloud = dataset.kitti_utils.get_point_cloud(point_cloud_source,
                                                      img_idx,
                                                      image_shape=image.shape)
    point_cloud = np.asarray(point_cloud)

    # Filter point cloud to extents
    area_extents = np.asarray([[-40, 40], [-5, 3], [0, 70]])
    bev_extents = area_extents[[0, 2]]

    points = point_cloud.T
    point_filter = obj_utils.get_point_filter(point_cloud, area_extents)
    points = points[point_filter]

    point_colours = vis_utils.project_img_to_point_cloud(points,
                                                         image,
                                                         dataset.calib_dir,
                                                         img_idx)

    # Voxelize the point cloud for visualization
    voxel_grid = VoxelGrid()
    voxel_grid.voxelize(points, voxel_size=0.1,
                        create_leaf_layout=False)

    # Ground plane
    ground_plane = obj_utils.get_road_plane(img_idx, dataset.planes_dir)

    ##############################
    # Visualization
    ##############################
    # Create VtkVoxelGrid
    vtk_voxel_grid = VtkVoxelGrid()
    vtk_voxel_grid.set_voxels(voxel_grid)

    vtk_point_cloud = VtkPointCloud()
    vtk_point_cloud.set_points(points, point_colours)

    # Create VtkAxes
    vtk_axes = vtk.vtkAxesActor()
    vtk_axes.SetTotalLength(5, 5, 5)

    # Create VtkBoxes for proposal boxes
    vtk_proposal_boxes = VtkBoxes()
    vtk_proposal_boxes.set_line_width(proposals_line_width)
    vtk_proposal_boxes.set_objects(proposal_objs,
                                   COLOUR_SCHEME_PREDICTIONS)

    # Create VtkBoxes for prediction boxes
    vtk_prediction_boxes = VtkPyramidBoxes()
    vtk_prediction_boxes.set_line_width(predictions_line_width)
    vtk_prediction_boxes.set_objects(prediction_objs,
                                     COLOUR_SCHEME_PREDICTIONS,
                                     show_orientations)

    # Create VtkBoxes for ground truth
    vtk_hard_gt_boxes = VtkBoxes()
    vtk_medium_gt_boxes = VtkBoxes()
    vtk_easy_gt_boxes = VtkBoxes()
    vtk_all_gt_boxes = VtkBoxes()

    vtk_hard_gt_boxes.set_objects(hard_gt_objs, COLOUR_SCHEME_PREDICTIONS,
                                  show_orientations)
    vtk_medium_gt_boxes.set_objects(medium_gt_objs, COLOUR_SCHEME_PREDICTIONS,
                                    show_orientations)
    vtk_easy_gt_boxes.set_objects(easy_gt_objs, COLOUR_SCHEME_PREDICTIONS,
                                  show_orientations)
    vtk_all_gt_boxes.set_objects(all_gt_objs, VtkBoxes.COLOUR_SCHEME_KITTI,
                                 show_orientations)

    # Create VtkTextLabels for 3D ious
    vtk_text_labels = VtkTextLabels()

    if draw_ious_3d and len(all_gt_boxes_3d) > 0:
        gt_positions_3d = np.asarray(all_gt_boxes_3d)[:, 0:3]
        vtk_text_labels.set_text_labels(
            gt_positions_3d,
            ['{:0.3f}'.format(iou_3d) for iou_3d in max_ious_3d])

    # Create VtkGroundPlane
    vtk_ground_plane = VtkGroundPlane()
    vtk_slice_bot_plane = VtkGroundPlane()
    vtk_slice_top_plane = VtkGroundPlane()

    vtk_ground_plane.set_plane(ground_plane, bev_extents)
    vtk_slice_bot_plane.set_plane(ground_plane + [0, 0, 0, -0.2], bev_extents)
    vtk_slice_top_plane.set_plane(ground_plane + [0, 0, 0, -2.0], bev_extents)

    # Create Voxel Grid Renderer in bottom half
    vtk_renderer = vtk.vtkRenderer()
    vtk_renderer.AddActor(vtk_voxel_grid.vtk_actor)
    vtk_renderer.AddActor(vtk_point_cloud.vtk_actor)

    vtk_renderer.AddActor(vtk_proposal_boxes.vtk_actor)
    vtk_renderer.AddActor(vtk_prediction_boxes.vtk_actor)

    vtk_renderer.AddActor(vtk_hard_gt_boxes.vtk_actor)
    vtk_renderer.AddActor(vtk_medium_gt_boxes.vtk_actor)
    vtk_renderer.AddActor(vtk_easy_gt_boxes.vtk_actor)
    vtk_renderer.AddActor(vtk_all_gt_boxes.vtk_actor)

    vtk_renderer.AddActor(vtk_text_labels.vtk_actor)

    # Add ground plane and slice planes
    vtk_renderer.AddActor(vtk_ground_plane.vtk_actor)
    vtk_renderer.AddActor(vtk_slice_bot_plane.vtk_actor)
    vtk_renderer.AddActor(vtk_slice_top_plane.vtk_actor)

    #vtk_renderer.AddActor(vtk_axes)
    vtk_renderer.SetBackground(0.2, 0.3, 0.4)

    # Set initial properties for some actors
    vtk_point_cloud.vtk_actor.GetProperty().SetPointSize(3)
    vtk_proposal_boxes.vtk_actor.SetVisibility(0)
    vtk_voxel_grid.vtk_actor.SetVisibility(0)
    vtk_all_gt_boxes.vtk_actor.SetVisibility(0)

    vtk_ground_plane.vtk_actor.SetVisibility(0)
    vtk_slice_bot_plane.vtk_actor.SetVisibility(0)
    vtk_slice_top_plane.vtk_actor.SetVisibility(0)
    vtk_ground_plane.vtk_actor.GetProperty().SetOpacity(0.9)
    vtk_slice_bot_plane.vtk_actor.GetProperty().SetOpacity(0.9)
    vtk_slice_top_plane.vtk_actor.GetProperty().SetOpacity(0.9)

    # Setup Camera
    current_cam = vtk_renderer.GetActiveCamera()
    current_cam.Pitch(160.0)
    current_cam.Roll(180.0)

    # Zooms out to fit all points on screen
    vtk_renderer.ResetCamera()
    # Zoom in slightly
    current_cam.Zoom(3.5)

    # Reset the clipping range to show all points
    vtk_renderer.ResetCameraClippingRange()

    # Setup Render Window
    vtk_render_window = vtk.vtkRenderWindow()
    vtk_render_window.SetWindowName(
        "Predictions: Step {}, Sample {}, Min Score {}".format(
            global_step,
            sample_name,
            mlod_score_threshold,
        ))
    vtk_render_window.SetSize(900, 600)
    vtk_render_window.AddRenderer(vtk_renderer)

    # Setup custom interactor style, which handles mouse and key events
    vtk_render_window_interactor = vtk.vtkRenderWindowInteractor()
    vtk_render_window_interactor.SetRenderWindow(vtk_render_window)

    # Add custom interactor to toggle actor visibilities
    custom_interactor = vis_utils.CameraInfoInteractorStyle([
        vtk_proposal_boxes.vtk_actor,
        vtk_prediction_boxes.vtk_actor,
        vtk_voxel_grid.vtk_actor,
        vtk_point_cloud.vtk_actor,

        vtk_easy_gt_boxes.vtk_actor,
        vtk_medium_gt_boxes.vtk_actor,
        vtk_hard_gt_boxes.vtk_actor,
        vtk_all_gt_boxes.vtk_actor,

        vtk_ground_plane.vtk_actor,
        vtk_slice_bot_plane.vtk_actor,
        vtk_slice_top_plane.vtk_actor,
        vtk_text_labels.vtk_actor,
    ])

    vtk_render_window_interactor.SetInteractorStyle(custom_interactor)
    # Render in VTK
    vtk_render_window.Render()

    # Take a screenshot
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(vtk_render_window)
    window_to_image_filter.Update()

    png_writer = vtk.vtkPNGWriter()
    file_name = img_out_dir + "/{}.png".format(sample_name)
    png_writer.SetFileName(file_name)
    png_writer.SetInputData(window_to_image_filter.GetOutput())
    png_writer.Write()

    print('Screenshot saved to ', file_name)

    vtk_render_window_interactor.Start()  # Blocking
    # vtk_render_window_interactor.Initialize()   # Non-Blocking


if __name__ == '__main__':
    main()
