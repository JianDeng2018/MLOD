import time

import cv2
import numpy as np
from wavedata.tools.core import calib_utils

from wavedata.tools.obj_detection import obj_utils

import mlod
from mlod.builders.dataset_builder import DatasetBuilder
from mlod.core.anchor_generators import grid_anchor_3d_generator
from mlod.datasets.kitti import kitti_aug


@profile
def main():

    # Create Dataset
    dataset_config_path = mlod.root_dir() + \
        '/configs/mb_preprocessing/rpn_cars.config'
    dataset = DatasetBuilder.load_dataset_from_config(dataset_config_path)

    # Random sample
    sample_name = '000169'

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

    flipped_points = flipped_point_cloud.T

    print('flip sample', time.time() - start_time)

    ##############################
    # Generate anchors
    ##############################
    clusters, _ = dataset.get_cluster_info()
    anchor_generator = grid_anchor_3d_generator.GridAnchor3dGenerator()

    # Read mini batch info
    anchors_info = dataset_utils.get_anchors_info(
        dataset.classes_name,
        anchor_strides,
        sample_name)

    all_anchor_boxes_3d = []
    all_ious = []
    for class_idx in range(len(dataset.classes)):

        anchor_boxes_3d = anchor_generator.generate(
            area_3d=dataset.kitti_utils.area_extents,
            anchor_3d_sizes=clusters[class_idx],
            anchor_stride=anchor_strides[class_idx],
            ground_plane=ground_plane)

        if anchors_info:
            indices, ious, offsets, classes = anchors_info

            # Get non empty anchors from the indices
            non_empty_anchor_boxes_3d = anchor_boxes_3d[indices]

            all_anchor_boxes_3d.extend(non_empty_anchor_boxes_3d)
            all_ious.extend(ious)

    if not len(all_anchor_boxes_3d) > 0:
        # Exit early if anchors_info is empty
        print("No anchors, Please try a different sample")
        return

    # Convert to ndarrays
    all_anchor_boxes_3d = np.asarray(all_anchor_boxes_3d)
    all_ious = np.asarray(all_ious)

    ##############################
    # Flip anchors
    ##############################
    start_time = time.time()

    flipped_anchor_boxes_3d = kitti_aug.flip_boxes_3d(all_anchor_boxes_3d,
                                                      flip_ry=False)

    print('flip anchors', time.time() - start_time)

    # Overwrite with flipped things
    all_anchor_boxes_3d = flipped_anchor_boxes_3d
    points = flipped_points
    ground_truth_list = flipped_gt_list
    ground_plane = flipped_ground_plane


if __name__ == '__main__':
    main()
