import time

import cv2
import numpy as np
import vtk

from wavedata.tools.core import calib_utils
from wavedata.tools.obj_detection import obj_utils
from wavedata.tools.visualization import vis_utils
from wavedata.tools.visualization.vtk_boxes import VtkBoxes
from wavedata.tools.visualization.vtk_point_cloud import VtkPointCloud

from mlod.builders.dataset_builder import DatasetBuilder
from mlod.datasets.kitti import kitti_aug


def project_flipped_img_to_point_cloud(points,
                                       image_flipped,
                                       calib_dir,
                                       img_idx):
    """ Projects image colours to point cloud points

    Arguments:
        points (N by [x,y,z]): list of points where N is
            the number of points
        image (Y by X by [r,g,b]): colour values in image space
        calib_dir (str): calibration directory
        img_idx (int): index of the requested image

    Returns:
        [N by [r,g,b]]: Matrix of colour codes. Indices of colours correspond
            to the indices of the points in the 'points' argument

    """
    # Save the pixel colour corresponding to each point
    frame_calib = calib_utils.read_calibration(calib_dir, img_idx)

    # Fix flipped p2 matrix
    flipped_p2 = np.copy(frame_calib.p2)
    flipped_p2[0, 2] = image_flipped.shape[1] - flipped_p2[0, 2]
    flipped_p2[0, 3] = -flipped_p2[0, 3]

    # Use fixed matrix
    point_in_im = calib_utils.project_to_image(
        points.T, p=flipped_p2).T

    point_in_im_rounded = np.floor(point_in_im)
    point_in_im_rounded = point_in_im_rounded.astype(np.int32)

    # image_shape = image_flipped.shape
    point_colours = []
    for point in point_in_im_rounded:
        point_colours.append(image_flipped[point[1], point[0], :])

    point_colours = np.asanyarray(point_colours)

    return point_colours


def main():
    """Shows a flipped sample in 3D
    """

    # Create Dataset
    dataset = DatasetBuilder.build_kitti_dataset(
        DatasetBuilder.KITTI_TRAINVAL)

    ##############################
    # Options
    ##############################
    # sample_name = "000191"
    sample_name = "000104"
    img_idx = int(sample_name)
    print("Showing anchors for sample {}".format(sample_name))

    ##############################
    # Load Sample Data
    ##############################
    ground_plane = obj_utils.get_road_plane(img_idx, dataset.planes_dir)

    image = cv2.imread(dataset.get_rgb_image_path(sample_name))
    image_shape = [image.shape[1], image.shape[0]]

    # Get point cloud
    point_cloud = obj_utils.get_depth_map_point_cloud(img_idx,
                                                      dataset.calib_dir,
                                                      dataset.depth_dir,
                                                      image_shape)

    points = np.array(point_cloud).T

    # Ground truth
    gt_labels = obj_utils.read_labels(dataset.label_dir, img_idx)

    # Filter ground truth
    gt_labels = dataset.kitti_utils.filter_labels(gt_labels)

    ##############################
    # Flip stuff
    ##############################
    image_flipped = np.fliplr(image)

    # Flip ground plane coeff (x)
    ground_plane_flipped = np.copy(ground_plane)
    ground_plane_flipped[0] = -ground_plane_flipped[0]

    # Flip 3D points
    points_flipped = kitti_aug.flip_points(points)

    # Get point cloud colours
    point_colours_flipped = project_flipped_img_to_point_cloud(
        points_flipped, image_flipped, dataset.calib_dir, img_idx)

    # Flip ground truth boxes
    gt_labels_flipped = [kitti_aug.flip_label_in_3d_only(obj)
                         for obj in gt_labels]

    ##############################
    # VTK Visualization
    ##############################
    # Axes
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(5, 5, 5)

    # Point cloud
    vtk_point_cloud = VtkPointCloud()
    vtk_point_cloud.set_points(points_flipped,
                               point_colours=point_colours_flipped)

    # # Ground Truth Boxes
    vtk_boxes = VtkBoxes()
    vtk_boxes.set_objects(gt_labels_flipped,
                          VtkBoxes.COLOUR_SCHEME_KITTI,
                          show_orientations=True)

    # Renderer
    vtk_renderer = vtk.vtkRenderer()
    vtk_renderer.SetBackground(0.2, 0.3, 0.4)

    # Add Actors to Rendered
    vtk_renderer.AddActor(axes)
    vtk_renderer.AddActor(vtk_point_cloud.vtk_actor)
    vtk_renderer.AddActor(vtk_boxes.vtk_actor)

    # Setup Camera
    current_cam = vtk_renderer.GetActiveCamera()
    current_cam.Pitch(170.0)
    current_cam.Roll(180.0)

    # Zooms out to fit all points on screen
    vtk_renderer.ResetCamera()

    # Zoom in slightly
    current_cam.Zoom(2.5)

    # Reset the clipping range to show all points
    vtk_renderer.ResetCameraClippingRange()

    # Setup Render Window
    vtk_render_window = vtk.vtkRenderWindow()
    vtk_render_window.SetWindowName("Anchors")
    vtk_render_window.SetSize(900, 500)
    vtk_render_window.AddRenderer(vtk_renderer)

    # Setup custom interactor style, which handles mouse and key events
    vtk_render_window_interactor = vtk.vtkRenderWindowInteractor()
    vtk_render_window_interactor.SetRenderWindow(vtk_render_window)
    vtk_render_window_interactor.SetInteractorStyle(
        vis_utils.ToggleActorsInteractorStyle([
            vtk_point_cloud.vtk_actor,
        ]))

    # Render in VTK
    vtk_render_window.Render()
    vtk_render_window_interactor.Start()


if __name__ == '__main__':
    main()
