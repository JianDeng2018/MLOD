import cv2
import sys
import time

import numpy as np

from wavedata.tools.core import calib_utils
from wavedata.tools.obj_detection import obj_utils

import mlod
from mlod.builders import config_builder_util
from mlod.builders.dataset_builder import DatasetBuilder

# Small hack to run even with @profile tags
import builtins
builtins.profile = lambda x: x


@profile
def main():

    test_pipeline_config_path = mlod.root_dir() + \
        '/data/configs/official/cars/cars_000_vanilla.config'
    model_config, train_config, _, dataset_config = \
        config_builder_util.get_configs_from_pipeline_file(
            test_pipeline_config_path, is_training=True)

    # train_val_test = 'val'
    # dataset_config.data_split = 'val'

    train_val_test = 'test'
    dataset_config.data_split = 'trainval'
    dataset_config.data_split_dir = 'training'
    dataset_config.has_labels = False

    # dataset_config.cache_config.cache_images = True
    # dataset_config.cache_config.cache_depth_maps = True

    dataset = DatasetBuilder.build_kitti_dataset(dataset_config,
                                                 use_defaults=False)
    kitti_utils = dataset.kitti_utils

    bev_source = 'lidar'
    # sample_name = '000000'
    # img_idx = np.random.randint(0, 1000)
    # sample_name = '{:06d}'.format(img_idx)

    num_samples = 200

    all_load_times = []
    all_bev_times = []
    for sample_idx in range(num_samples):
        sys.stdout.write(
            '\rSample {} / {}'.format(sample_idx, num_samples - 1))

        img_idx = sample_idx
        sample_name = '{:06d}'.format(sample_idx)

        loading_start_time = time.time()
        # Load image
        image = cv2.imread(dataset.get_rgb_image_path(sample_name))
        image_shape = image.shape[0:2]
        calib_p2 = calib_utils.read_calibration(dataset.calib_dir, img_idx)

        point_cloud = kitti_utils.get_point_cloud(bev_source,
                                                  int(sample_name),
                                                  image_shape)
        ground_plane = kitti_utils.get_ground_plane(sample_name)
        all_load_times.append(time.time() - loading_start_time)

        bev_start_time = time.time()
        bev_maps = kitti_utils.create_bev_maps(point_cloud, ground_plane)
        bev_end_time = time.time()
        all_bev_times.append(bev_end_time - bev_start_time)

    print('')
    print('Load mean:', np.mean(all_load_times))
    print('Load median:', np.median(all_load_times))
    print('BEV mean:', np.mean(all_bev_times))
    print('BEV median:', np.median(all_bev_times))


if __name__ == '__main__':
    main()
