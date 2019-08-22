import copy

import numpy as np

from wavedata.tools.obj_detection import data_aug
from wavedata.tools.core import calib_utils
import math

AUG_FLIPPING = 'flipping'
AUG_PCA_JITTER = 'pca_jitter'
AUG_RANDOM_OCC = 'occ_mask'

perms = [(0, 1, 2), (0, 2, 1),
        (1, 0, 2), (1, 2, 0),
        (2, 0, 1), (2, 1, 0)]


def flip_image(image):
    """Flips an image horizontally
    """
    flipped_image = np.fliplr(image)
    return flipped_image


def flip_points(points):
    """Flips a list of points (N, 3)
    """
    flipped_points = np.copy(points)
    flipped_points[:, 0] = -points[:, 0]
    return flipped_points


def flip_point_cloud(point_cloud):
    """Flips a point cloud (3, N)
    """
    flipped_point_cloud = np.copy(point_cloud)
    flipped_point_cloud[0] = -point_cloud[0]
    return flipped_point_cloud


def flip_label_in_3d_only(obj_label):
    """Flips only the 3D position of an object label. The 2D bounding box is
    not flipped to save time since it is not used.

    Args:
        obj_label: ObjectLabel

    Returns:
        A flipped object
    """

    flipped_label = copy.deepcopy(obj_label)

    # Flip the rotation
    if obj_label.ry >= 0:
        flipped_label.ry = np.pi - obj_label.ry
    else:
        flipped_label.ry = -np.pi - obj_label.ry

    # Flip the t.x sign, t.y and t.z remains the unchanged
    flipped_t = (-flipped_label.t[0], flipped_label.t[1], flipped_label.t[2])
    flipped_label.t = flipped_t

    return flipped_label

def flip_label(obj_label,image_shape):
    """Flips the 2D/3D position of an object label. 

    Args:
        obj_label: ObjectLabel
        image_shape: (height,width)

    Returns:
        A flipped object
    """
    flipped_label = copy.deepcopy(obj_label)
    #Flip the 2D label
    flipped_label.x1 = image_shape[1]-obj_label.x2
    flipped_label.x2 = image_shape[1]-obj_label.x1

     # Flip the rotation
    if obj_label.ry >= 0:
        flipped_label.ry = np.pi - obj_label.ry
    else:
        flipped_label.ry = -np.pi - obj_label.ry

    # Flip the t.x sign, t.y and t.z remains the unchanged
    flipped_t = (-flipped_label.t[0], flipped_label.t[1], flipped_label.t[2])
    flipped_label.t = flipped_t

    return flipped_label

def flip_boxes_3d(boxes_3d, flip_ry=True):
    """Flips boxes_3d

    Args:
        boxes_3d: List of boxes in box_3d format
        flip_ry bool: (optional) if False, rotation is not flipped to save on
            computation (useful for flipping anchors)

    Returns:
        flipped_boxes_3d: Flipped boxes in box_3d format
    """

    flipped_boxes_3d = np.copy(boxes_3d)

    if flip_ry:
        # Flip the rotation
        above_zero = boxes_3d[:, 6] >= 0
        below_zero = np.logical_not(above_zero)
        flipped_boxes_3d[above_zero, 6] = np.pi - boxes_3d[above_zero, 6]
        flipped_boxes_3d[below_zero, 6] = -np.pi - boxes_3d[below_zero, 6]

    # Flip the t.x sign, t.y and t.z remains the unchanged
    flipped_boxes_3d[:, 0] = -boxes_3d[:, 0]

    return flipped_boxes_3d


def flip_ground_plane(ground_plane):
    """Flips the ground plane by negating the x coefficient
        (ax + by + cz + d = 0)

    Args:
        ground_plane: ground plane coefficients

    Returns:
        Flipped ground plane coefficients
    """
    flipped_ground_plane = np.copy(ground_plane)
    flipped_ground_plane[0] = -ground_plane[0]
    return flipped_ground_plane


def flip_stereo_calib_p2(calib_p2, image_shape):
    """Flips the stereo calibration matrix to correct the projection back to
    image space. Flipping the image can be seen as a movement of both the
    camera plane, and the camera itself. To account for this, the instrinsic
    matrix x0 value is flipped with respect to the image width, and the
    extrinsic matrix t1 value is negated.

    Args:
        calib_p2: 3 x 4 stereo camera calibration matrix
        image_shape: (h, w) image shape

    Returns:
        'Flipped' calibration p2 matrix with shape (3, 4)
    """
    flipped_p2 = np.copy(calib_p2)
    flipped_p2[0, 2] = image_shape[1] - calib_p2[0, 2]
    flipped_p2[0, 3] = -calib_p2[0, 3]

    return flipped_p2


def apply_pca_jitter(image_in, aug_img_noise=False):
    """Applies PCA jitter or random noise to a single image

    Args:
        image_in: Image to modify
        aug_img_noise (bool): If True, add random augmentation

    Returns:
        Modified image
    """
    image_out = np.asarray([image_in])

    if not aug_img_noise:
        pca = data_aug.compute_pca(image_in)
        image_out = data_aug.add_pca_jitter(image_in, pca)

    else:
        # Random value
        if np.random.randint(2):
            # PCA Jitter
            pca = data_aug.compute_pca(image_out)
            image_out = data_aug.add_pca_jitter(image_out, pca)

        if np.random.randint(2):
            # Swap channels in RGB
            random_perm = perms[np.random.choice(6)]
            image_out = image_out[:,:,:,random_perm]

        if np.random.randint(2):
            # random Contrast
            alpha = np.random.uniform(0.5, 1.5)
            alpha = np.uint8(alpha)
            image_out *= alpha
            image_out = np.clip(image_out,0.0, 255.0)

        if np.random.randint(2):
            # Brightness
            delta = np.random.uniform(-32.0, 32.0)
            delta = np.uint8(delta)
            image_out += delta
            image_out = np.clip(image_out, 0.0, 255.0)

    image_out = image_out.astype(dtype=np.uint8)

    return image_out

def occ_aug(point_cloud, calib, labels):
    # point cloud
    # calib: calibration matrix
    # masks labels list 
    # return augumentated point cloud

    point_in_im = calib_utils.project_to_image(point_cloud, p=calib).T
    point_cloud = point_cloud.T
    mask_labels = occ_aug_mask(labels)

    occ_filter = False

    for obj in mask_labels:
        occ_filter = occ_filter | (point_in_im[:, 0] > obj[0]) & \
                     (point_in_im[:, 0] < obj[2]) & \
                     (point_in_im[:, 1] > obj[1]) & \
                     (point_in_im[:, 1] < obj[3])

    return point_cloud[np.logical_not(occ_filter)].T

def occ_aug_mask(labels):

    mask_label = []
    for obj in labels:
        x_min = int(min(obj.x1, obj.x2))
        x_max = int(max(obj.x2,obj.x1))
        y_min = math.ceil(obj.y1)
        y_max = math.floor(obj.y2)
        w_max = int(0.5*(x_max-x_min)/2)
        if x_min >= x_max:
            #It happens in sample 3840, 1 pixel wideth object
            continue
        x0 =np.random.randint(x_min,x_max,size=1)
        y0 =np.random.randint(y_min,y_max,size=1)
        w = w_max
        mask_label.append([x0-w,y0,x0+w,y_max])
 
    return mask_label

