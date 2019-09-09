"""
Metrics evaluation logic for Free Space Estimation
"""

import cv2
import numpy as np


def metrics(cv2_seg, cv2_gt, data_class):
    """
    Given 2 opencv2 images of same size, this function
    calculates 3 metrics for evaluating success of SegNet resultant:
        - 1) global pixel-wise accuracy across all classes
        - 2) mean average of all class-specific accuracies
        - 3) class-wise mean Intersection over Union (mIoU)
        - category-wise mIoU not currently implemented
        - loop structure is set up to allow category-wise mIoU
        - instance-level IoU will require some rework

    :param cv2_seg:     SegNet resultant opencv2 image
    :param cv2_gt:      ground truth opencv2 image
    :param data_class:  data semantics ("eval_data_cityscape.py" is reference)
    :return:            global average (G), class accuracy average (C), mIoU
    """

    assert cv2_seg.shape == cv2_gt.shape, "images must have same dimensions"
    c_sum, miou_sum = np.array([]), np.array([])

    # ignore 'void' group/category
    for grp in data_class[:-1]:
        for elem in grp:
            # elem[0] == class name along the given axis befo
            # elem[1] == BGR
            # elem[2] == seg_full
            # elem[3] == truth_full
            # elem[4] == true_pos
            # elem[5] == false_pos
            # elem[6] == false_negs

            # store pixel masks for class 'elem' in ground truth and seg frame
            elem[2] = cv2.inRange(cv2_seg, elem[1], elem[1])
            elem[3] = cv2.inRange(cv2_gt, elem[1], elem[1])

            not_full_seg = cv2.bitwise_not(elem[2])
            not_full_gt = cv2.bitwise_not(elem[3])

            # store masks for true positives, false positives, false negatives
            elem[4] = cv2.bitwise_and(elem[2], elem[2], mask=elem[3])
            elem[5] = cv2.bitwise_and(elem[2], elem[2], mask=not_full_gt)
            elem[6] = cv2.bitwise_and(elem[3], elem[3], mask=not_full_seg)

            # calculate class accuracy if relevant
            num_gt = cv2.countNonZero(elem[3])
            num_tp = cv2.countNonZero(elem[4]) + 0.
            if num_gt > 0:
                c_sum = np.append(c_sum, num_tp / (num_gt + 0.))
            else:
                c_sum = np.append(c_sum, np.nan)

            # calculate IoU if relevant
            iou_denominator = num_tp + \
                cv2.countNonZero(elem[5]) + cv2.countNonZero(elem[6])
            if iou_denominator > 0:
                miou_sum = np.append(miou_sum, num_tp / (iou_denominator + 0.))
            else:
                miou_sum = np.append(miou_sum, np.nan)

    # global average (G) == equivalent pixels / total pixels
    g_avg = (cv2_seg == cv2_gt).sum() / (cv2_gt.size + 0.)

    return g_avg, c_sum, miou_sum
