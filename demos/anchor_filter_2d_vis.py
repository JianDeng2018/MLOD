import numpy as np
import vtk
import time

from wavedata.tools.core import voxel_grid_2d
from wavedata.tools.core import voxel_grid
from wavedata.tools.obj_detection import obj_utils
from wavedata.tools.visualization import vis_utils
from wavedata.tools.visualization.vtk_boxes import VtkBoxes
from wavedata.tools.visualization.vtk_voxel_grid import VtkVoxelGrid

from mlod.builders.dataset_builder import DatasetBuilder
from mlod.core import box_3d_encoder
from mlod.core.anchor_generators import grid_anchor_3d_generator
from mlod.core import anchor_filter


def main():
    """
    Visualization for comparison of anchor filtering with
        2D vs 3D integral images

    Keys:
        F1: Toggle 3D integral image filtered anchors
        F2: Toggle 2D integral image filtered anchors
        F3: Toggle 2D integral image empty anchors
    """

    anchor_2d_colour_scheme = {"Anchor": (0, 0, 255)}  # Blue
    anchor_3d_colour_scheme = {"Anchor": (0, 255, 0)}  # Green
    anchor_unfiltered_colour_scheme = {"Anchor": (255, 0, 255)}  # Purple

    # Create Dataset
    dataset = DatasetBuilder.build_kitti_dataset(
        DatasetBuilder.KITTI_TRAINVAL)

    sample_name = "000001"
    img_idx = int(sample_name)
    print("Showing anchors for sample {}".format(sample_name))

    # Options
    # These clusters are from the trainval set and give more 2D anchors than 3D
    clusters = np.array([[3.55, 1.835, 1.525], [4.173, 1.69, 1.49]])
    anchor_stride = [3.0, 3.0]

    ground_plane = obj_utils.get_road_plane(img_idx, dataset.planes_dir)
    area_extents = np.array([[-40, 40], [-5, 3], [0, 70]])

    anchor_3d_generator = grid_anchor_3d_generator.GridAnchor3dGenerator()

    # Generate anchors
    start_time = time.time()
    anchor_boxes_3d = anchor_3d_generator.generate(area_3d=area_extents,
                                                   anchor_3d_sizes=clusters,
                                                   anchor_stride=anchor_stride,
                                                   ground_plane=ground_plane)
    end_time = time.time()
    print("Anchors generated in {} s".format(end_time - start_time))

    # Get point cloud
    point_cloud = obj_utils.get_stereo_point_cloud(img_idx, dataset.calib_dir,
                                                   dataset.disp_dir)

    ground_offset_dist = 0.2
    offset_dist = 2.0

    # Filter points within certain xyz range and offset from ground plane
    # Filter points within 0.2m of the road plane
    slice_filter = dataset.kitti_utils.create_slice_filter(point_cloud,
                                                           area_extents,
                                                           ground_plane,
                                                           ground_offset_dist,
                                                           offset_dist)
    points = np.array(point_cloud).T
    points = points[slice_filter]

    anchors = box_3d_encoder.box_3d_to_anchor(anchor_boxes_3d)

    # Create 2D voxel grid
    vx_grid_2d = voxel_grid_2d.VoxelGrid2D()
    vx_grid_2d.voxelize_2d(points, 0.1, area_extents)

    # Create 3D voxel grid
    vx_grid_3d = voxel_grid.VoxelGrid()
    vx_grid_3d.voxelize(points, 0.1, area_extents)

    # Filter the boxes here!
    start_time = time.time()
    empty_filter_2d = anchor_filter.get_empty_anchor_filter_2d(
        anchors=anchors,
        voxel_grid_2d=vx_grid_2d,
        density_threshold=1)
    anchors_2d = anchor_boxes_3d[empty_filter_2d]
    end_time = time.time()
    print("2D Anchors filtered in {} s".format(end_time - start_time))
    print("Number of 2D anchors remaining: %d" % (anchors_2d.shape[0]))

    unfiltered_anchors_2d = anchor_boxes_3d[np.logical_not(empty_filter_2d)]

    # 3D filtering
    start_time = time.time()
    empty_filter_3d = anchor_filter.get_empty_anchor_filter(
        anchors=anchors,
        voxel_grid_3d=vx_grid_3d,
        density_threshold=1)
    anchor_boxes_3d = anchor_boxes_3d[empty_filter_3d]
    end_time = time.time()
    print("3D Anchors filtered in {} s".format(end_time - start_time))
    print("Number of 3D anchors remaining: %d" % (anchor_boxes_3d.shape[0]))

    anchor_2d_objects = []
    for anchor_idx in range(len(anchors_2d)):
        anchor = anchors_2d[anchor_idx]
        obj_label = box_3d_encoder.box_3d_to_object_label(anchor, 'Anchor')

        # Append to a list for visualization in VTK later
        anchor_2d_objects.append(obj_label)

    anchor_3d_objects = []
    for anchor_idx in range(len(anchor_boxes_3d)):
        anchor = anchor_boxes_3d[anchor_idx]
        obj_label = box_3d_encoder.box_3d_to_object_label(anchor, 'Anchor')

        # Append to a list for visualization in VTK later
        anchor_3d_objects.append(obj_label)

    unfiltered_anchor_objects = []
    for anchor_idx in range(len(unfiltered_anchors_2d)):
        anchor = unfiltered_anchors_2d[anchor_idx]
        obj_label = box_3d_encoder.box_3d_to_object_label(anchor, 'Anchor')

        # Append to a list for visualization in VTK later
        unfiltered_anchor_objects.append(obj_label)

    # Create VtkAxes
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(5, 5, 5)

    # Create VtkBoxes for boxes
    vtk_2d_boxes = VtkBoxes()
    vtk_2d_boxes.set_objects(anchor_2d_objects, anchor_2d_colour_scheme)

    vtk_3d_boxes = VtkBoxes()
    vtk_3d_boxes.set_objects(anchor_3d_objects, anchor_3d_colour_scheme)

    vtk_unfiltered_boxes = VtkBoxes()
    vtk_unfiltered_boxes.set_objects(unfiltered_anchor_objects,
                                     anchor_unfiltered_colour_scheme)

    vtk_voxel_grid = VtkVoxelGrid()
    vtk_voxel_grid.set_voxels(vx_grid_3d)

    vtk_voxel_grid_2d = VtkVoxelGrid()
    vtk_voxel_grid_2d.set_voxels(vx_grid_2d)

    # Create Voxel Grid Renderer in bottom half
    vtk_renderer = vtk.vtkRenderer()
    vtk_renderer.AddActor(vtk_2d_boxes.vtk_actor)
    vtk_renderer.AddActor(vtk_3d_boxes.vtk_actor)
    vtk_renderer.AddActor(vtk_unfiltered_boxes.vtk_actor)
    vtk_renderer.AddActor(vtk_voxel_grid.vtk_actor)
    vtk_renderer.AddActor(vtk_voxel_grid_2d.vtk_actor)
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
        vis_utils.ToggleActorsInteractorStyle([
            vtk_2d_boxes.vtk_actor,
            vtk_3d_boxes.vtk_actor,
            vtk_unfiltered_boxes.vtk_actor,
        ]))

    # Render in VTK
    vtk_render_window.Render()
    vtk_render_window_interactor.Start()


if __name__ == '__main__':
    main()
