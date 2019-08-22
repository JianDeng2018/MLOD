import numpy as np
import time
import vtk

from PIL import Image

import matplotlib.pyplot as plt

from wavedata.tools.core import voxel_grid
from wavedata.tools.obj_detection import obj_utils
from wavedata.tools.visualization import vis_utils
from wavedata.tools.visualization.vtk_boxes import VtkBoxes
from wavedata.tools.visualization.vtk_point_cloud import VtkPointCloud
from wavedata.tools.visualization.vtk_voxel_grid import VtkVoxelGrid

from mlod.core import anchor_projector
from mlod.core import box_3d_encoder
from mlod.builders.dataset_builder import DatasetBuilder
from mlod.core.anchor_generators import grid_anchor_3d_generator
from mlod.core import anchor_filter


def main():
    """
    Visualization of anchor filtering using 3D integral images
    """

    anchor_colour_scheme = {"Car": (0, 255, 0),  # Green
                            "Pedestrian": (255, 150, 50),  # Orange
                            "Cyclist": (150, 50, 100),  # Purple
                            "DontCare": (255, 0, 0),  # Red
                            "Anchor": (0, 0, 255),  # Blue
                            }

    # Create Dataset
    dataset = DatasetBuilder.build_kitti_dataset(DatasetBuilder.KITTI_TRAINVAL)

    # Options
    clusters, _ = dataset.get_cluster_info()
    sample_name = "000000"
    img_idx = int(sample_name)
    anchor_stride = [0.5, 0.5]
    ground_plane = obj_utils.get_road_plane(img_idx, dataset.planes_dir)

    anchor_3d_generator = grid_anchor_3d_generator.GridAnchor3dGenerator(
        anchor_3d_sizes=clusters,
        anchor_stride=anchor_stride
    )

    area_extents = np.array([[-40, 40], [-5, 3], [0, 70]])

    # Generate anchors in box_3d format
    start_time = time.time()
    anchor_boxes_3d = anchor_3d_generator.generate(area_3d=area_extents,
                                                   ground_plane=ground_plane)
    end_time = time.time()
    print("Anchors generated in {} s".format(end_time - start_time))

    point_cloud = obj_utils.get_lidar_point_cloud(img_idx, dataset.calib_dir,
                                                  dataset.velo_dir)

    offset_dist = 2.0

    # Filter points within certain xyz range and offset from ground plane
    offset_filter = obj_utils.get_point_filter(point_cloud, area_extents,
                                               ground_plane, offset_dist)

    # Filter points within 0.2m of the road plane
    road_filter = obj_utils.get_point_filter(point_cloud, area_extents,
                                             ground_plane, 0.1)

    slice_filter = np.logical_xor(offset_filter, road_filter)
    point_cloud = point_cloud.T[slice_filter]

    # Generate Voxel Grid
    vx_grid_3d = voxel_grid.VoxelGrid()
    vx_grid_3d.voxelize(point_cloud, 0.1, area_extents)

    # Anchors in anchor format
    all_anchors = box_3d_encoder.box_3d_to_anchor(anchor_boxes_3d)

    # Filter the boxes here!
    start_time = time.time()
    empty_filter = \
        anchor_filter.get_empty_anchor_filter(anchors=all_anchors,
                                              voxel_grid_3d=vx_grid_3d,
                                              density_threshold=1)
    anchor_boxes_3d = anchor_boxes_3d[empty_filter]
    end_time = time.time()
    print("Anchors filtered in {} s".format(end_time - start_time))

    # Visualize GT boxes
    # Grab ground truth
    ground_truth_list = obj_utils.read_labels(dataset.label_dir, img_idx)

    # ----------
    # Test Sample extraction

    # Visualize from here
    vis_utils.visualization(dataset.rgb_image_dir, img_idx)
    plt.show(block=False)

    image_path = dataset.get_rgb_image_path(sample_name)
    image_shape = np.array(Image.open(image_path)).shape
    rgb_boxes, rgb_normalized_boxes = \
        anchor_projector.project_to_image_space(all_anchors, dataset,
                                                image_shape, img_idx)

    # Overlay boxes on images
    anchor_objects = []
    for anchor_idx in range(len(anchor_boxes_3d)):
        anchor_box_3d = anchor_boxes_3d[anchor_idx]
        obj_label = box_3d_encoder.box_3d_to_object_label(anchor_box_3d,
                                                          'Anchor')
        # Append to a list for visualization in VTK later
        anchor_objects.append(obj_label)

    for idx in range(len(ground_truth_list)):
        ground_truth_obj = ground_truth_list[idx]
        # Append to a list for visualization in VTK later
        anchor_objects.append(ground_truth_obj)

    # Create VtkAxes
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(5, 5, 5)

    # Create VtkBoxes for boxes
    vtk_boxes = VtkBoxes()
    vtk_boxes.set_objects(anchor_objects, anchor_colour_scheme)

    vtk_point_cloud = VtkPointCloud()
    vtk_point_cloud.set_points(point_cloud)

    vtk_voxel_grid = VtkVoxelGrid()
    vtk_voxel_grid.set_voxels(vx_grid_3d)

    # Create Voxel Grid Renderer in bottom half
    vtk_renderer = vtk.vtkRenderer()
    vtk_renderer.AddActor(vtk_boxes.vtk_actor)
    # vtk_renderer.AddActor(vtk_point_cloud.vtk_actor)
    vtk_renderer.AddActor(vtk_voxel_grid.vtk_actor)
    vtk_renderer.AddActor(axes)
    vtk_renderer.SetBackground(0.2, 0.3, 0.4)

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
        vtk.vtkInteractorStyleTrackballCamera())

    # Render in VTK
    vtk_render_window.Render()
    vtk_render_window_interactor.Start()  # Blocking


if __name__ == '__main__':
    main()
