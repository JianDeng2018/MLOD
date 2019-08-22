import numpy as np
import time
import vtk

from wavedata.tools.core import voxel_grid
from wavedata.tools.visualization.vtk_boxes import VtkBoxes
from wavedata.tools.visualization.vtk_point_cloud import VtkPointCloud
from wavedata.tools.visualization.vtk_voxel_grid import VtkVoxelGrid

from mlod.builders.dataset_builder import DatasetBuilder
from mlod.core import box_3d_encoder
from mlod.core.label_cluster_utils import LabelClusterUtils

from mlod.core import anchor_filter


def main():
    """
    Simple demo script for debugging integral images with visualization
    """
    anchor_colour_scheme = {"Anchor": (0, 0, 255)}  # Blue

    dataset = DatasetBuilder.build_kitti_dataset(DatasetBuilder.KITTI_TRAIN)

    label_cluster_utils = LabelClusterUtils(dataset)
    clusters, _ = label_cluster_utils.get_clusters()

    area_extents = np.array([[0, 2], [-1, 0.], [0, 2]])
    boxes_3d = np.array([
        [2, 0, 1, 1, 1, 1, 0],
        [1, 0, 2, 1, 1, 1, 0],
    ])

    xyz = np.array([[0.5, -0.01, 1.1],
                    [1.5, -0.01, 1.1],
                    [0.5, -0.01, 1.6],
                    [1.5, -0.01, 1.6],
                    [0.5, -0.49, 1.1],
                    [1.5, -0.49, 1.1],
                    [0.5, -0.51, 1.6],
                    [1.5, -0.51, 1.6]
                    ])

    vx_grid_3d = voxel_grid.VoxelGrid()
    vx_grid_3d.voxelize(xyz, 0.1, area_extents)

    anchors = box_3d_encoder.box_3d_to_anchor(boxes_3d)

    # Filter the boxes here!
    start_time = time.time()
    empty_filter = anchor_filter.get_empty_anchor_filter(anchors=anchors,
                                                         voxel_grid_3d=vx_grid_3d,
                                                         density_threshold=1)
    boxes_3d = boxes_3d[empty_filter]
    end_time = time.time()
    print("Anchors filtered in {} s".format(end_time - start_time))

    box_objects = []
    for box_idx in range(len(boxes_3d)):
        box = boxes_3d[box_idx]
        obj_label = box_3d_encoder.box_3d_to_object_label(box, 'Anchor')

        # Append to a list for visualization in VTK later
        box_objects.append(obj_label)

    # Create VtkAxes
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(5, 5, 5)

    # Create VtkBoxes for boxes
    vtk_boxes = VtkBoxes()
    vtk_boxes.set_objects(box_objects, anchor_colour_scheme)

    vtk_point_cloud = VtkPointCloud()
    vtk_point_cloud.set_points(xyz)

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
    vtk_render_window_interactor.Start()


if __name__ == '__main__':
    main()
