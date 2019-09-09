#!usr/bin/env python
import numpy as np
from wavedata.tools.core import calib_utils


def pc_to_xyz(point_cloud,
              calib_p,
              image_shape):
    """Projects Lidar Point cloud to XYZ image encoding.

    :param point_cloud: input point cloud (3,N)
    :param calib_p: stereo calibration p2 matrix
    :param image_shape: image shape [h, w]
    :return xyz_enc: point cloud encoded as an (3 x h x w) with x y z
    """

    num_of_points = len(point_cloud.T)

    point_in_im = calib_utils.project_to_image(point_cloud, p=calib_p).T

    point_in_im_rounded = np.int32(np.floor(point_in_im))

    all_x, all_y, all_z = fill_xyz(point_cloud,
                                   image_shape,
                                   num_of_points,
                                   point_in_im_rounded)

    xyz_enc = np.dstack((all_x, all_y, all_z))

    return xyz_enc


def collision_check(point_cloud,
                    image_shape,
                    num_of_points,
                    point_in_im_rounded):
    """Provides an array of the same size as the image
    encoded with the indices of points in the point cloud

    :param point_cloud: input point cloud (3,N)
    :param image_shape: image shape [h, w]
    :param num_of_points: size of the point cloud
    :param point_in_im_rounded: coordinates of each point in image space
    :return all_indices: indices of the closest points to the image center
                            (h x w x 1)
    """

    all_indices = np.full(image_shape, np.inf)

    valid_ind = [point_in_im_rounded[:, 1], point_in_im_rounded[:, 0]]

    distances_from_camera = point_cloud[0, :] ** 2 + \
                            point_cloud[1, :] ** 2 + \
                            point_cloud[2, :] ** 2

    all_indices[valid_ind] = [idx for idx in range(num_of_points)
                              if all_indices[point_in_im_rounded[idx, 1],
                                             point_in_im_rounded[idx, 0]]
                              >= distances_from_camera[idx]]
    return all_indices


def fill_xyz(point_cloud,
             image_shape,
             num_of_points,
             point_in_im_rounded):
    """Encode three arrays with the same size of the image
       with the x, y, z coordinates. Empty pixels are filled with NaN

    :param point_cloud: input point cloud (3,N)
    :param image_shape: image shape [h, w]
    :param num_of_points: size of the point cloud
    :param point_in_im_rounded: coordinates of each point in image space
    :return all_indices: indices of the closest points to the image center
                        (h x w x 1)
    """

    # Fill in x, y and z images with indeces from collision checker
    all_x = np.full(image_shape, np.NaN)
    all_y = np.full(image_shape, np.NaN)
    all_z = np.full(image_shape, np.NaN)

    all_indices = collision_check(point_cloud,
                                  image_shape,
                                  num_of_points,
                                  point_in_im_rounded)

    for ind in range(num_of_points):

        idx_current = np.int_(all_indices[point_in_im_rounded[ind, 1],
                                          point_in_im_rounded[ind, 0]])

        if idx_current != np.inf:
            mat_ind = [point_in_im_rounded[idx_current, 1],
                       point_in_im_rounded[idx_current, 0]]

            all_x[mat_ind[0], mat_ind[1]] = point_cloud[0, idx_current]

            all_y[mat_ind[0], mat_ind[1]] = point_cloud[1, idx_current]

            all_z[mat_ind[0], mat_ind[1]] = point_cloud[2, idx_current]

    return all_x, all_y, all_z
