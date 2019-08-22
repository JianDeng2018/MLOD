import os

import cv2
import numpy as np
import vtk

from wavedata.tools.core.voxel_grid import VoxelGrid
from wavedata.tools.obj_detection import obj_utils
from wavedata.tools.visualization.vtk_boxes import VtkBoxes
from wavedata.tools.visualization.vtk_box_4c import VtkBox4c
from wavedata.tools.visualization.vtk_voxel_grid import VtkVoxelGrid
from wavedata.tools.visualization.vtk_point_cloud import VtkPointCloud
from wavedata.tools.visualization import vis_utils

import mlod
from mlod.builders.dataset_builder import DatasetBuilder
from mlod.utils import demo_utils


from mlod.core import box_3d_encoder


def main():
    """This demo visualizes box 8C format predicted by MLOD, before
    getting converted to Box 3D.

    Keys:
        F1: Toggle predictions
        F2: Toggle easy ground truth objects (Green)
        F3: Toggle medium ground truth objects (Orange)
        F4: Toggle hard ground truth objects (Red)
        F5: Toggle all ground truth objects (default off)

        F6: Toggle 3D voxel grid
        F7: Toggle point cloud
    """
    ##############################
    # Options
    ##############################
    mlod_score_threshold = 0.1
    show_orientations = True

    checkpoint_name = 'mlod_exp_example'

    global_step = None
    global_step = 100000

    sample_name = None

    # # # Cars # # #
    sample_name = '000050'
    # sample_name = '000104'
    # sample_name = '000169'
    # sample_name = '000175'
    # sample_name = '000191'
    # sample_name = '000335'
    # sample_name = '000360'
    # sample_name = '001783'
    # sample_name = '001820'
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

    dataset_config = DatasetBuilder.copy_config(
        DatasetBuilder.KITTI_VAL_HALF)

    dataset = DatasetBuilder.build_kitti_dataset(dataset_config)
    ##############################
    # Setup Paths
    ##############################

    # Random sample
    if sample_name is None:
        sample_idx = np.random.randint(0, dataset.num_samples)
        sample_name = dataset.sample_list[sample_idx]

    img_idx = int(sample_name)

    # Text files directory
    prediction_boxes_3d_dir = mlod.root_dir() + \
        '/data/outputs/' + checkpoint_name + '/predictions' +  \
        '/final_predictions_and_scores/' + dataset.data_split

    prediction_boxes_4c_dir = mlod.root_dir() + \
        '/data/outputs/' + checkpoint_name + '/predictions' +  \
        '/final_boxes_4c_and_scores/' + dataset.data_split

    # Get checkpoint step
    steps = os.listdir(prediction_boxes_3d_dir)
    steps.sort(key=int)
    print('Available steps: {}'.format(steps))

    # Use latest checkpoint if no index provided
    if global_step is None:
        global_step = steps[-1]

    ##############################
    # Load Predictions
    ##############################
    # Load prediction boxes_3d from files
    prediction_boxes_3d_and_scores = np.loadtxt(
        prediction_boxes_3d_dir + "/{}/{}.txt".format(
            global_step, sample_name))

    pred_boxes_3d = prediction_boxes_3d_and_scores[:, 0:7]
    # pred_boxes_3d_scores = prediction_boxes_3d_and_scores[:, 8]

    # Load prediction boxes_4c
    prediction_boxes_4c_and_scores = np.loadtxt(
        prediction_boxes_4c_dir + "/{}/{}.txt".format(
            global_step, sample_name))
    pred_boxes_4c = prediction_boxes_4c_and_scores[:, 0:10]
    pred_boxes_4c_scores = prediction_boxes_4c_and_scores[:, 10]

    # Filter by score
    score_mask = pred_boxes_4c_scores >= mlod_score_threshold
    pred_boxes_3d = pred_boxes_3d[score_mask]
    pred_boxes_4c = pred_boxes_4c[score_mask]

    ground_plane = obj_utils.get_road_plane(img_idx, dataset.planes_dir)

    # Visualization
    all_vtk_boxes_4c = []
    for box_4c in pred_boxes_4c:
        vtk_box_4c = VtkBox4c()
        vtk_box_4c.set_box_4c(box_4c, ground_plane)
        all_vtk_boxes_4c.append(vtk_box_4c)

    # Convert boxes_3d to ObjectLabels
    pred_objs = np.asarray(
        [box_3d_encoder.box_3d_to_object_label(box_3d, obj_type='Car')
         for box_3d in pred_boxes_3d])
    vtk_boxes_3d = VtkBoxes()
    vtk_boxes_3d.set_objects(pred_objs, VtkBoxes.COLOUR_SCHEME_KITTI,
                             show_orientations=True)

    ##############################
    # Ground Truth
    ##############################
    if dataset.has_labels:
        easy_gt_objs, medium_gt_objs, \
            hard_gt_objs, all_gt_objs = \
            demo_utils.get_gts_based_on_difficulty(dataset,
                                                   img_idx)
    else:
        easy_gt_objs = medium_gt_objs = hard_gt_objs = all_gt_objs = []

    ##############################
    # Point Cloud
    ##############################
    image_path = dataset.get_rgb_image_path(sample_name)
    image = cv2.imread(image_path)
    img_idx = int(sample_name)

    points, point_colours = demo_utils.get_filtered_pc_and_colours(dataset,
                                                                   image,
                                                                   img_idx)

    # # Voxelize the point cloud for visualization
    # voxel_grid = VoxelGrid()
    # voxel_grid.voxelize(points, voxel_size=0.1,
    #                     create_leaf_layout=False)

    ##############################
    # Visualization
    ##############################
    # # Create VtkVoxelGrid
    # vtk_voxel_grid = VtkVoxelGrid()
    # vtk_voxel_grid.set_voxels(voxel_grid)

    vtk_point_cloud = VtkPointCloud()
    vtk_point_cloud.set_points(points, point_colours)

    # Create VtkAxes
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(5, 5, 5)

    # Create VtkBoxes for ground truth
    vtk_easy_gt_boxes, vtk_medium_gt_boxes, \
        vtk_hard_gt_boxes, vtk_all_gt_boxes = \
        demo_utils.create_gt_vtk_boxes(easy_gt_objs,
                                       medium_gt_objs,
                                       hard_gt_objs,
                                       all_gt_objs,
                                       show_orientations)

    # Create Voxel Grid Renderer in bottom half
    vtk_renderer = vtk.vtkRenderer()
    vtk_renderer.SetBackground(0.2, 0.3, 0.4)

    vtk_renderer.AddActor(vtk_point_cloud.vtk_actor)

    vtk_box_actors = vtk.vtkAssembly()

    # Prediction boxes
    for i in range(len(all_vtk_boxes_4c)):
        # Adding labels, slows down rendering
        vtk_renderer.AddActor(all_vtk_boxes_4c[i].vtk_text_labels.vtk_actor)
        vtk_box_actors.AddPart(all_vtk_boxes_4c[i].vtk_actor)

    vtk_renderer.AddActor(vtk_boxes_3d.vtk_actor)

    vtk_renderer.AddActor(vtk_point_cloud.vtk_actor)

    vtk_renderer.AddActor(vtk_hard_gt_boxes.vtk_actor)
    vtk_renderer.AddActor(vtk_medium_gt_boxes.vtk_actor)
    vtk_renderer.AddActor(vtk_easy_gt_boxes.vtk_actor)
    vtk_renderer.AddActor(vtk_all_gt_boxes.vtk_actor)

    vtk_renderer.AddActor(vtk_box_actors)

    vtk_renderer.AddActor(axes)

    # Set initial properties for some actors
    vtk_point_cloud.vtk_actor.GetProperty().SetPointSize(2)

    vtk_all_gt_boxes.vtk_actor.SetVisibility(0)
    vtk_boxes_3d.vtk_actor.SetVisibility(0)

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

    vtk_render_window_interactor.SetInteractorStyle(
        vis_utils.ToggleActorsInteractorStyle([
            vtk_box_actors,
            vtk_boxes_3d.vtk_actor,

            vtk_easy_gt_boxes.vtk_actor,
            vtk_medium_gt_boxes.vtk_actor,
            vtk_hard_gt_boxes.vtk_actor,
            vtk_all_gt_boxes.vtk_actor,

            vtk_point_cloud.vtk_actor,
        ]))

    vtk_render_window_interactor.Start()


if __name__ == '__main__':
    main()
