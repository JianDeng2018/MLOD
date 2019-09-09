import collections

import cv2
import numpy as np
import png

from wavedata.tools.core import calib_utils

FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)


def interpolate(point_cloud, calib_p2, image_shape, custom_kernel, max_depth,
                show_process=False):
    """Interpolates the lidar point cloud by projecting the points into image
    space, and then applying classical image processing steps. The projected
    image is dilated with a custom kernel, closed, and a bilateral blur is
    applied to smooth the final output depth map.

    :param point_cloud: input point cloud (N, 3)
    :param calib_p2: stereo calibration p2 matrix
    :param image_shape: image shape [h, w]
    :param custom_kernel: custom kernel for initial dilation
    :param max_depth: maximum output depth
    :param show_process: (optional) flag to return image processing steps

    :return final_depths: interpolated lidar depth map
    :return process_dict: if show_process is True, this is an OrderedDict
        with entries showing the image processing steps, None otherwise
    """

    all_points = point_cloud.T

    # Save the depth corresponding to each point
    point_in_im = calib_utils.project_to_image(
        all_points.T, p=calib_p2).T
    point_in_im_rounded = np.int32(np.floor(point_in_im))

    # Invert depths
    all_points[:, 2] = max_depth - all_points[:, 2]

    # Vectorized version
    all_depths = np.zeros(image_shape)
    valid_indices = [point_in_im_rounded[:, 1], point_in_im_rounded[:, 0]]
    all_depths[valid_indices] = [max(
        all_depths[point_in_im_rounded[idx, 1], point_in_im_rounded[idx, 0]],
        all_points[idx, 2])
        for idx in range(len(all_points))]

    # Loop version (obsolete, keeping to show logic and as a backup)
    # all_depths = np.zeros((image_shape[1], image_shape[0]))
    # for point_idx in range(len(point_in_im_rounded)):
    #     map_x = point_in_im_rounded[point_idx, 1]
    #     map_y = point_in_im_rounded[point_idx, 0]
    #
    #     point_depth = all_points[point_idx, 2]
    #
    #     # Keep the farther distance for overlapping points
    #     if all_depths[map_x, map_y] > 0.0:
    #         all_depths[map_x, map_y] = \
    #             max(all_depths[map_x, map_y], point_depth)
    #     else:
    #         all_depths[map_x, map_y] = point_depth
    #
    #     # Clip to specified maximum depth
    #     all_depths[map_x, map_y] = \
    #         np.minimum(all_depths[map_x, map_y], max_depth)

    # Fill in the depth map
    lidar_depths = np.float32(all_depths)

    # Operations
    depths_in = lidar_depths
    dilated_depths = cv2.dilate(depths_in, custom_kernel)
    closed_depths = cv2.morphologyEx(dilated_depths,
                                     cv2.MORPH_CLOSE,
                                     FULL_KERNEL_5)
    blurred_depths = cv2.bilateralFilter(closed_depths, 3, 1, 2)
    depths_out = blurred_depths

    # Save final version to final_depths variable to be used later
    final_depths = depths_out.copy()

    # Invert
    valid_pixels = np.where(final_depths > 0.5)
    valid_pixels = np.asarray(valid_pixels).T
    final_depths[valid_pixels[:, 0], valid_pixels[:, 1]] = \
        max_depth - final_depths[valid_pixels[:, 0], valid_pixels[:, 1]]

    process_dict = None
    if show_process:
        process_dict = collections.OrderedDict()
        process_dict['lidar_depths'] = lidar_depths

        process_dict['dilated_depths'] = dilated_depths
        process_dict['closed_depths'] = closed_depths
        process_dict['blurred_depths'] = blurred_depths

        process_dict['final_depths'] = final_depths

    return final_depths, process_dict


def project_depths(point_cloud, camera_p, image_shape, max_depth=100.0):
    """Projects a point cloud into image space and saves depths per pixel.

    :param point_cloud: point cloud (N, 3)
    :param camera_p: stereo calibration p matrix
    :param image_shape: image shape [h, w]
    :param max_depth: image shape [h, w]

    :return all_depths: projected depth map
    """

    # Only keep points in front of the camera
    point_cloud = point_cloud.T
    #point_cloud = point_cloud[point_cloud[:,0] > 0]

    # Save the depth corresponding to each point
    point_in_im = calib_utils.project_to_image(point_cloud.T, p=camera_p).T
    point_in_im_rounded = np.array(np.int32(np.floor(point_in_im)))

    #filtered out out of boxes points
    image_filter = (point_in_im_rounded[:, 0] > 0) & \
                       (point_in_im_rounded[:, 0] < image_shape[1]) & \
                       (point_in_im_rounded[:, 1] > 0) & \
                       (point_in_im_rounded[:, 1] < image_shape[0])
    point_in_im_rounded = point_in_im_rounded[image_filter]    
    
    all_points = np.array(point_cloud)

    # Invert depths
    all_points[:, 2] = max_depth - all_points[:, 2]

    # Only save valid pixels, keep closer points when overlapping
    projected_depths = np.zeros(image_shape)
    x_proj = np.zeros(image_shape)
    y_proj = np.zeros(image_shape)
    valid_indices = tuple([point_in_im_rounded[:, 1], point_in_im_rounded[:, 0]])

    projected_depths[valid_indices] = [
        max(projected_depths[
            point_in_im_rounded[idx, 1], point_in_im_rounded[idx, 0]],
            all_points[idx, 2])
        for idx in range(len(point_in_im_rounded))]

    projected_depths[valid_indices] = \
        max_depth - projected_depths[valid_indices]

    x_proj[valid_indices] = [all_points[idx, 0]
        for idx in range(len(point_in_im_rounded))]

    y_proj[valid_indices] = [all_points[idx, 1]
        for idx in range(len(point_in_im_rounded))]

    return projected_depths

def project_depths_xy(point_cloud, camera_p, image_shape, max_depth=100.0):
    """Projects a point cloud into image space and saves depths per pixel.

    :param point_cloud: point cloud (N, 3)
    :param camera_p: stereo calibration p matrix
    :param image_shape: image shape [h, w]
    :param max_depth: image shape [h, w]

    :return all_depths: projected depth map
    """

    # Only keep points in front of the camera
    point_cloud = point_cloud.T
    #point_cloud = point_cloud[point_cloud[:,0] > 0]

    # Save the depth corresponding to each point
    point_in_im = calib_utils.project_to_image(point_cloud.T, p=camera_p).T
    point_in_im_rounded = np.array(np.int32(np.floor(point_in_im)))

    #filtered out out of boxes points
    image_filter = (point_in_im_rounded[:, 0] > 0) & \
                       (point_in_im_rounded[:, 0] < image_shape[1]) & \
                       (point_in_im_rounded[:, 1] > 0) & \
                       (point_in_im_rounded[:, 1] < image_shape[0])
    point_in_im_rounded = point_in_im_rounded[image_filter]    
    
    all_points = np.array(point_cloud)

    # Invert depths
    all_points[:, 2] = max_depth - all_points[:, 2]

    # Only save valid pixels, keep closer points when overlapping
    projected_depths = np.zeros(image_shape)
    x_proj = np.zeros(image_shape)
    y_proj = np.zeros(image_shape)
    valid_indices = tuple([point_in_im_rounded[:, 1], point_in_im_rounded[:, 0]])

    projected_depths[valid_indices] = [
        max(projected_depths[
            point_in_im_rounded[idx, 1], point_in_im_rounded[idx, 0]],
            all_points[idx, 2])
        for idx in range(len(point_in_im_rounded))]

    projected_depths[valid_indices] = \
        max_depth - projected_depths[valid_indices]

    x_proj[valid_indices] = [all_points[idx, 0]
        for idx in range(len(point_in_im_rounded))]

    y_proj[valid_indices] = [all_points[idx, 1]
        for idx in range(len(point_in_im_rounded))]

    return projected_depths,x_proj,y_proj


def fill_in_fast(depth_map, max_depth=100.0, custom_kernel=CROSS_KERNEL_5,
                 extend_to_top=False, blur_type='bilateral'):
    """Modified fast version of IP-Basic depth completion, done in place.

    :param  depth_map: projected depths
    :param  max_depth: max depth value for inversion
    :param  custom_kernel: kernel to apply initial dilation
    :param  extend_to_top: whether to extend depths to top of the frame
    :param  blur_type: 'bilateral' or 'gaussian'

    :return depth_map: denser depth map
    """

    # Convert to float32
    depth_map = np.asarray(depth_map, np.float32)

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # Dilate
    depth_map = cv2.dilate(depth_map, custom_kernel)

    # Median blur
    depth_map = cv2.medianBlur(depth_map, 5)

    # Hole closing
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Fill empty spaces with dilated values
    empty_pixels = (depth_map < 0.1)
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image
    if extend_to_top:
        for pixel_col_idx in range(depth_map.shape[1]):
            pixel_col = depth_map[:, pixel_col_idx]
            top_pixel_row = np.argmax(pixel_col > 0.1)
            top_pixel_value = depth_map[top_pixel_row, pixel_col_idx]
            depth_map[0:top_pixel_row, pixel_col_idx] = top_pixel_value

        # Large hole fill
        empty_pixels = depth_map < 0.1
        dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
        depth_map[empty_pixels] = dilated[empty_pixels]

    # Median blur
    depth_map = cv2.medianBlur(depth_map, 5)

    # Bilateral or Gaussian blur
    if blur_type == 'bilateral':
        # Bilateral blur
        depth_map = cv2.bilateralFilter(depth_map, 5, 1, 2)
    elif blur_type == 'gaussian':
        # Gaussian blur
        valid_pixels = (depth_map > 0.1)
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    return depth_map


def save_to_file(file_path, depth_map):
    """Saves a depth map as a uint16 png

    :param file_path: file path
    :param depth_map: depth map
    """

    with open(file_path, 'wb') as f:
        depth_image = (depth_map * 256).astype(np.uint16)

        # pypng is used because cv2 cannot save uint16 format images
        writer = png.Writer(width=depth_image.shape[1],
                            height=depth_image.shape[0],
                            bitdepth=16,
                            greyscale=True)
        writer.write(f, depth_image)


def get_depth_map(img_idx, depth_dir):
    """Reads the depth map from the depth directory

    :param img_idx: image index
    :param depth_dir: directory with depth maps

    :returns: ndarray of depths
    """
    # Get point cloud from depth map
    depth_file_path = depth_dir + '/{:06d}.png'.format(img_idx)
    depth_image = cv2.imread(depth_file_path, cv2.IMREAD_ANYDEPTH)

    if depth_image is not None:
        depth_map = depth_image / 256.0
        return depth_map

    raise FileNotFoundError('Please generate depth maps first.')
