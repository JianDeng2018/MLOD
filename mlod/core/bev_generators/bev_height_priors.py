import numpy as np

from wavedata.tools.core.voxel_grid_2d import VoxelGrid2D

from mlod.core.bev_generators import bev_generator


def gaussian_prior(x, means, std_devs, std_dev_multiplier):
    """Remaps x using a gaussian distribution. Creates a map
      for each set of means and standard deviations provided

    Args:
        x: ndarray input map
        means: array of means
        std_devs: array of standard deviations
        std_dev_multiplier: multiplier to control sparseness of maps

    Returns:
        An ndarray map for each set of means and standard deviations
    """
    maps = []

    for cluster_idx in range(len(means)):
        u = means[cluster_idx]
        s = np.array(std_devs[cluster_idx]) * std_dev_multiplier

        # Set standard deviation to 0.1 if std dev is 0 (only 1 instance)
        s = np.clip(s, a_min=0.1, a_max=float("inf"))

        # Return output between 0 and 1
        gaussian_map = 1.0 * np.exp(-(x - u) ** 2 / (2 * (s ** 2)))
        maps.append(gaussian_map)

    return maps


class BevHeightPriors(bev_generator.BevGenerator):

    LOG_16 = np.log(16)
    LOG_32 = np.log(32)
    LOG_64 = np.log(64)

    NORM_VALUES = {
        'stereo': LOG_64,
        'lidar': LOG_16,
        'depth': LOG_32,
    }

    def __init__(self, config, kitti_utils):
        """BEV maps created using gaussian height priors.

        Args:
            config: bev_generator protobuf config
            kitti_utils: KittiUtils object
        """

        # Parse config
        self.ground_filter_offset = config.ground_filter_offset
        self.offset_filter_distance = config.offset_filter_distance
        self.std_dev_multiplier = config.std_dev_multiplier

        self.kitti_utils = kitti_utils

    def generate_bev(self,
                     source,
                     point_cloud,
                     ground_plane,
                     area_extents,
                     voxel_size):
        """Generates the BEV maps dictionary. One height map is created for
        each cluster size in the dataset. One density map is created for
        the whole point cloud.

        Args:
            source: point cloud source, only used for normalization
            point_cloud: point cloud (3, N)
            ground_plane: ground plane coefficients
            area_extents: 3D area extents
                [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
            voxel_size: voxel size in m

        Returns:
            BEV maps dictionary
                height_maps: list of height maps
                density_map: density map
        """

        slice_filter = self.kitti_utils.create_slice_filter(
            point_cloud,
            area_extents,
            ground_plane,
            self.ground_filter_offset,
            self.offset_filter_distance)

        # Reshape points into N x [x, y, z]
        all_points = np.transpose(point_cloud)[slice_filter]

        # Create Voxel Grid 2D
        voxel_grid_2d = VoxelGrid2D()
        voxel_grid_2d.voxelize_2d(
            all_points, voxel_size,
            extents=area_extents,
            ground_plane=ground_plane,
            create_leaf_layout=False)

        # Remove y values (all 0)
        voxel_indices_2d = voxel_grid_2d.voxel_indices[:, [0, 2]]

        all_clusters = np.concatenate(self.kitti_utils.clusters)
        all_std_devs = np.concatenate(self.kitti_utils.std_devs)

        # Create empty BEV images
        height_maps = np.zeros((len(all_clusters),
                                voxel_grid_2d.num_divisions[0],
                                voxel_grid_2d.num_divisions[2]))

        # Get last element of clusters (l, w, h)
        height_clusters = np.vstack(all_clusters)[:, [2]]
        height_std_devs = np.vstack(all_std_devs)[:, [2]]

        height_priors = gaussian_prior(voxel_grid_2d.heights,
                                       height_clusters, height_std_devs,
                                       self.std_dev_multiplier)

        # Only update pixels where voxels have max height values
        height_maps[:, voxel_indices_2d[:, 0], voxel_indices_2d[:, 1]] = \
            np.asarray(height_priors)

        # Rotate images 90 degrees
        # (transpose and flip) is faster than np.rot90
        height_maps_out = [np.flip(height_maps[map_idx].transpose(), axis=0)
                           for map_idx in range(len(height_maps))]

        # Generate density map
        density_map = self._create_density_map(
            num_divisions=voxel_grid_2d.num_divisions,
            voxel_indices_2d=voxel_indices_2d,
            num_pts_per_voxel=voxel_grid_2d.num_pts_in_voxel,
            norm_value=self.NORM_VALUES[source])

        bev_maps = dict()
        bev_maps['height_maps'] = height_maps_out
        bev_maps['density_map'] = density_map

        return bev_maps
