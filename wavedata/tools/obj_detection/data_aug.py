""" Data Augmentation utilities
"""

import numpy as np
import random
import math
import copy
import cv2
from copy import deepcopy

from wavedata.tools.obj_detection import evaluation
from wavedata.tools.obj_detection import obj_utils as od


def compute_pca(image_set):
    """
    Calculate and returns PCA of a set of images

    :param image_set: List of images read with cv2.imread in np.uint8 format
    :return: PCA for the set of images
    """

    # Check for valid input
    assert(image_set[0].dtype == np.uint8)

    # Reshape data into single array
    reshaped_data = np.concatenate([image
                                    for pixels in image_set for image in
                                    pixels])

    # Convert to float and normalize the data between [0, 1]
    reshaped_data = (reshaped_data / 255.0).astype(np.float32)

    # Calculate covariance, eigenvalues, and eigenvectors
    # np.cov calculates covariance around the mean, so no need to shift the
    # data
    covariance = np.cov(reshaped_data.T)
    e_vals, e_vecs = np.linalg.eigh(covariance)

    # svd can also be used instead
    # U, S, V = np.linalg.svd(mean_data)

    pca = np.sqrt(e_vals) * e_vecs

    return pca


def add_pca_jitter(img_data, pca):
    """
    Adds a multiple of the principle components,
    with magnitude from a Gaussian distribution with mean 0 and stdev 0.1

    :param img_data: Original image in read with cv2.imread in np.uint8 format
    :param pca: PCA calculated with compute_PCA for the image set

    :return: Image with added noise
    """

    # Check for valid input
    assert (img_data.dtype == np.uint8)

    # Make a copy of the image data
    new_img_data = np.copy(img_data).astype(np.float32) / 255.0

    # Calculate noise by multiplying pca with magnitude,
    # then sum horizontally since eigenvectors are in columns
    magnitude = np.random.randn(3) * 0.1
    noise = (pca * magnitude).sum(axis=1)

    # Add the noise to the image, and clip to valid range [0, 1]
    new_img_data = new_img_data + noise
    np.clip(new_img_data, 0.0, 1.0, out=new_img_data)

    # Change back to np.uint8
    new_img_data = (new_img_data * 255).astype(np.uint8)

    return new_img_data


def flip_label(obj_label, im_size):
    """Flips an object label along x

    :param obj_label: ObjectLabel
    :param im_size: (w, h) image size
    :return: Flipped ObjectLabel
    """

    flipped_label = copy.deepcopy(obj_label)

    # Flip in 2D
    x1 = flipped_label.x1
    x2 = flipped_label.x2

    half_width = im_size[0] / 2.0

    diff = x1 - half_width

    # width of bounding box
    width_bb = x2 - x1

    if x1 < half_width:
        new_x2 = half_width + abs(diff)
    else:
        new_x2 = half_width - abs(diff)
    new_x1 = new_x2 - width_bb

    # since we are doing mirror flip,
    # the y's remain unchanged
    flipped_label.x1 = int(new_x1)
    flipped_label.x2 = int(new_x2)

    # Flip in 3D

    # Flip the rotation (mirror effect)
    # the angle pointing into the camera is pi/2
    half_pi = math.pi / 2
    ry_diff = half_pi - flipped_label.ry
    flipped_label.ry = half_pi + ry_diff

    # Flip the t.x sign, t.y and t.z remains the unchanged
    flipped_t = (-flipped_label.t[0], flipped_label.t[1], flipped_label.t[2])
    flipped_label.t = flipped_t

    return flipped_label


def flip_2d_image_and_label(image, label):
    """Flips 2D image from left to right.

    The image is flipped using OpenCV's flip matrix
    function. The flipped 2D label is then calculated
    by splitting the image in half and deciding which
    split it belongs and adding the offset to align
    the bounding box.

    Arguments:
        image: cv2.Matrix image
        label: np.array of 4 bounding box corners

    Returns:
        flipped_image: ndarray flipped image
        flipped_label: ObjectLabel to match the flipped
        object inside bounding box.
    """
    image_flipped = cv2.flip(image, 1)

    # Flip the label
    width = image.shape[1]
    half_width = width / 2

    flipped_label = np.copy(label)

    x1 = label[0]
    x2 = label[2]

    diff = x1 - half_width
    # width of bounding box
    width_bb = x2 - x1
    assert(x2 > x1)

    if x1 < half_width:
        new_x2 = half_width + abs(diff)
    else:
        new_x2 = half_width - abs(diff)
    new_x = new_x2 - width_bb

    # since we are doing mirror flip,
    # the y's remain unchanged
    flipped_label[0] = int(new_x)
    flipped_label[2] = int(new_x2)

    return image_flipped, flipped_label


def flip_3d_point_and_label(label, points=None):
    """Flips 3D point cloud and bounding box.

    This is a mirror effect. For point cloud,
    we flip along the x-axis. For the label,
    everything stays the same (l, w, h etc)
    the only thing that changes is the centroid
    and orientation. For that, we just need
    to flip the signs to create the mirror
    effect.

    Keyword Arguments:
        label: ObjectLabel describing the
               location of the bounding box
               containing an object.
        points: (optional) list of ndarray
                including x, y, z points.
                This is optional as its more
                convenient for testing purposes.

    Returns:
        flipped_label: ObjectLabel representing
                      the flipped position of the
                      bounding box.
        flipped_points: list of ndarray containing
                        flipped point clouds. Returned
                        only of points are provided.
    """

    # flip the label
    flipped_label = deepcopy(label)
    # flip the rotation (mirror effect)
    # the angle pointing into the camera is pi/2
    half_pi = math.pi / 2
    ry_diff = half_pi - label.ry
    flipped_label.ry = half_pi + ry_diff
    # flip the t.x sign, t.y and t.z remains the unchanged
    flipped_t = (-label.t[0], label.t[1], label.t[2])
    flipped_label.t = flipped_t

    # flip the points along x-coordinates
    if points is not None:
        xp = points[0]
        # flip the x's by negating them
        x_flipped = [-x for x in xp]
        flipped_points = [x_flipped, points[1], points[2]]
        return flipped_label, flipped_points
    else:
        return flipped_label


def calculate_negative_2d_bb(obj_label,
                             boxes2d,
                             iou_threshold_min,
                             iou_threshold_max,
                             samples,
                             rand_sampl=False):
    """Generates negative 2D bounding boxes.

    This is computed in a semi-brute force fashion.
    For any given bounding box, we first try to calculate
    the desired shift based on the selected IoU. If this
    failes, we just randomly shift the centroid and generate
    new bounding boxes. If it lies within the desired IoU
    threshold bound, we will keep it. Otherwise it is thrown
    out and this is repeated until number of valid samples is
    satisfied.

    Keyword arguments:
        obj_label: single label for a detected 2D object
        boxes2d: a list of numpy array representing all
                 bounding boxes in the image
        iou_threshold_min: determines the min variation
            between the original iou and the negative samples
        iou_threshold_max: determines the max variation
            between the original iou and the negative samples
        samples: number of negative samples per detected object
        rand_sampl: Flag to switch between calculation vs pure
                    random sampling. For speed testing purposes.

    Returns:
        new_objects: a list of randomly generated ObjectLabels
        failed_cases: int number of cases it failed to calculate
                 and opted in for random sampling.
    """

    x1 = obj_label.x1
    y1 = obj_label.y1
    x2 = obj_label.x2
    y2 = obj_label.y2

    width = x2 - x1
    length = y2 - y1

    half_w = width / 2
    half_l = length / 2

    miscalc = False
    current_samples = 0
    new_objects = []

    failed_cases = 0

    while current_samples < samples:
        # Keep trying to generate samples that
        # lie within reasonable bound

        if not miscalc and not rand_sampl:
            # we will try to to this by calculating
            # the shift along l or w given the desired
            # IoU and fixing either w or l shift.
            # this is sub-optimal since it does not
            # guarantee to achive the actual desired
            # IoU since we keep one variable fixed!

            # First let's try to generate bounding box
            # by randomly selecting a desirable IoU
            possible_ious = np.linspace(iou_threshold_min,
                                        iou_threshold_max,
                                        10)
            # pick one randomly
            rand_iou = random.choice(possible_ious)

            # assuming l and w here are equal, given IoU
            # by fixing either delta_w or delta_l we can
            # calculate the other. This way we *guess*
            # the generated box will satify the IoU bound
            # constraint

            l_fixed = random.choice([True, False])
            if l_fixed:
                # Lets keep delta_l fixed
                # It just needs to keep it within 0 - l
                delta_l = random.uniform(0, length / 2)
                l_shift = length - delta_l
                w_shift = (rand_iou * length * width) / (l_shift)
                delta_w = width - w_shift
            else:
                # keep delta_w fixed
                delta_w = random.uniform(0, width / 2)
                w_shift = length - delta_w
                l_shift = (rand_iou * length * width) / (w_shift)
                delta_l = length - l_shift

            # now just shift the l and w
            new_xp = x1 + delta_w
            new_yp = y1 + delta_l
        else:
            # that didn't work in the previous iteration
            # try generating random points
            new_xp = np.random.uniform(x1, x2, 1)
            new_yp = np.random.uniform(y1, y2, 1)
            # give it another chance in the next iteration
            miscalc = False

        new_obj, new_box = _construct_new_2d_object(new_xp,
                                                    half_w,
                                                    new_yp,
                                                    half_l)

        # calculate the IoU
        iou = evaluation.two_d_iou(new_box, boxes2d)
        # check if it generated the desired IoU
        if iou_threshold_min < max(iou) < iou_threshold_max:
            # keep the new object label
            current_samples += 1
            new_objects.append(new_obj)
        else:
            failed_cases += 1
            miscalc = True

    return new_objects, failed_cases


def _construct_new_2d_object(new_xp,
                             half_w,
                             new_yp,
                             half_l):
    """Helper function to construct a
       new object label and prepare
       arguments to calculate IoU. Used
       inside generate_negative_2d_bb

    """

    new_x1 = float(new_xp - half_w)
    new_x2 = float(new_xp + half_w)
    new_y1 = float(new_yp - half_l)
    new_y2 = float(new_yp + half_l)

    new_obj = od.ObjectLabel()
    new_obj.x1 = new_x1
    new_obj.x2 = new_x2
    new_obj.y1 = new_y1
    new_obj.y2 = new_y2

    new_box = np.array([new_x1, new_y1, new_x2, new_y2])

    return new_obj, new_box


def generate_negative_2d_bb(obj_label,
                            boxes2d,
                            iou_threshold_min,
                            iou_threshold_max,
                            samples):
    """Generates negative 2D bounding boxes.
    This is computed in a brute force fashion. For any given
    bounding box, we randomly shift the centroid and generate
    new bounding boxes and if it lies within the desired IoU
    threshold bound, we will keep it. Otherwise it is thrown
    out and this is repeated until number of valid samples is
    satisfied.
    Keyword arguments:
        obj_label: single label for a detected 2D object
        boxes2d: a list of numpy array representing all
            bounding boxes in the image
        iou_threshold_min: determines the min variation
            between the original iou and the negative samples
        iou_threshold_max: determines the max variation
            between the original iou and the negative samples
        samples: number of negative samples per detected object
    Returns: a list of generated ObjectLabels
    """

    x1 = obj_label.x1
    y1 = obj_label.y1
    x2 = obj_label.x2
    y2 = obj_label.y2

    diff_x = (x2 - x1) / 2
    diff_y = (y2 - y1) / 2

    current_samples = 0
    new_objects = []

    while current_samples < samples:
        # Keep trying to generate samples that
        # lie within reasonable bound
        new_xp = np.random.uniform(x1, x2, 1)
        new_yp = np.random.uniform(y1, y2, 1)

        new_x1 = float(new_xp - diff_x)
        new_x2 = float(new_xp + diff_x)
        new_y1 = float(new_yp - diff_y)
        new_y2 = float(new_yp + diff_y)

        new_obj = od.ObjectLabel()
        new_obj.x1 = new_x1
        new_obj.x2 = new_x2
        new_obj.y1 = new_y1
        new_obj.y2 = new_y2

        new_box = np.array([new_x1, new_y1, new_x2, new_y2])

        # calculate the IoU
        iou = evaluation.two_d_iou(new_box, boxes2d)

        if iou_threshold_min < max(iou) < iou_threshold_max:
            # keep the new object label
            current_samples += 1
            new_objects.append(new_obj)

    return new_objects


def generate_negative_3d_bb(obj_label,
                            boxes3d,
                            iou_threshold_min,
                            iou_threshold_max,
                            samples):
    """Generates negative 3D bounding boxes.

    This is the 3D version of generate_negative_3d_bb.

    Keyword arguments:
        obj_label: single label for a detected 3D object
        boxes3d: a list of numpy array representing all
                 3D bounding boxes in the image
        iou_threshold_min: determines the min variation
            between the original iou and the negative samples
        iou_threshold_max: determines the max variation
            between the original iou and the negative samples
        samples: number of negative samples per detected object

    Returns: a list of generated ObjectLabels
    """
    box_corners = od.compute_box_corners_3d(obj_label)
    # make sure this is not empty
    assert(len(box_corners) > 0)
    P1 = box_corners[:, 0]
    P2 = box_corners[:, 1]
    P4 = box_corners[:, 3]

    current_samples = 0
    new_objects = []

    while current_samples < samples:
        # Generate random 3D point inside the box
        # we keep the same y, only generate x and z
        new_xp = float(np.random.uniform(P1[0], P4[0], 1))
        new_zp = float(np.random.uniform(P1[2], P2[2], 1))

        # create a new ObjectLabel
        new_obj = copy.copy(obj_label)
        # update the new obj.t
        # everything else is the same, only changing the
        # centroid point t which remains the same along
        # the y direction
        new_obj.t = (new_xp, obj_label.t[1], new_zp)

        _, box_to_test, _ = od.build_bbs_from_objects([new_obj], 'All')

        assert(len(box_to_test) == 1)
        # we are dealing with one box here so its the first element
        iou_3d = evaluation.three_d_iou(box_to_test[0], boxes3d)

        # check if iou_3d is a list, take the largest
        # this compares to all the boxes
        if isinstance(iou_3d, np.ndarray):
            iou_max = np.max(iou_3d)
        else:
            iou_max = iou_3d

        if iou_threshold_min < iou_max < iou_threshold_max:
            # keep the new object label
            new_objects.append(new_obj)
            current_samples += 1

    return new_objects
