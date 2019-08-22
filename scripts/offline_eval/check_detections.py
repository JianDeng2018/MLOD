import sys

import matplotlib.pyplot as plt
import numpy as np
from wavedata.tools.obj_detection import obj_utils, evaluation

from mlod.builders.dataset_builder import DatasetBuilder
from mlod.core import box_3d_encoder


def main():
    """Plots detection errors for xyz, lwh, ry, and shows 3D IoU with
    ground truth boxes
    """

    dataset = DatasetBuilder.build_kitti_dataset(DatasetBuilder.KITTI_VAL_HALF,
                                                 use_defaults=True)

    difficulty = 2

    # Full path to kitti predictions
    # (e.g. '.../data/outputs/mlod_exp_example/predictions/'
    # 'kitti_predictions_3d/val/0.1/100000/data'
    predictions_data_path = 'path_to_detections/data'

    # Loop through split and save ious and errors
    all_3d_ious = []
    all_errors = []

    for sample_idx in range(dataset.num_samples):

        sys.stdout.write('\r{} / {}'.format(
            sample_idx + 1, dataset.num_samples))

        sample_name = dataset.sample_names[sample_idx]
        img_idx = int(sample_name)

        # Get filtered ground truth
        all_gt_objs = obj_utils.read_labels(dataset.label_dir, img_idx)
        all_gt_objs = dataset.kitti_utils.filter_labels(
            all_gt_objs, difficulty=difficulty)

        pred_objs = obj_utils.read_labels(predictions_data_path, img_idx)

        ##############################
        # Check IoUs
        ##############################
        if len(all_gt_objs) > 0 and \
                pred_objs is not None and len(pred_objs) > 0:

            all_gt_boxes_3d = [box_3d_encoder.object_label_to_box_3d(gt_obj)
                               for gt_obj in all_gt_objs]
            pred_boxes_3d = [box_3d_encoder.object_label_to_box_3d(pred_obj)
                             for pred_obj in pred_objs]

            # Convert to iou format
            gt_objs_iou_fmt = box_3d_encoder.box_3d_to_3d_iou_format(
                all_gt_boxes_3d)
            pred_objs_iou_fmt = box_3d_encoder.box_3d_to_3d_iou_format(
                pred_boxes_3d)

            max_ious_3d = np.zeros(len(all_gt_objs))
            max_iou_pred_indices = -np.ones(len(all_gt_objs))
            for gt_obj_idx in range(len(all_gt_objs)):
                gt_obj_iou_fmt = gt_objs_iou_fmt[gt_obj_idx]

                ious_3d = evaluation.three_d_iou(gt_obj_iou_fmt,
                                                 pred_objs_iou_fmt)

                max_iou_3d = np.amax(ious_3d)
                max_ious_3d[gt_obj_idx] = max_iou_3d

                if max_iou_3d > 0.0:
                    max_iou_pred_indices[gt_obj_idx] = np.argmax(ious_3d)

            for gt_obj_idx in range(len(all_gt_objs)):

                max_iou_pred_idx = int(max_iou_pred_indices[gt_obj_idx])
                if max_iou_pred_idx >= 0:
                    error = all_gt_boxes_3d[gt_obj_idx] - \
                            pred_boxes_3d[max_iou_pred_idx]

                    all_errors.append(error)

            all_3d_ious.extend(max_ious_3d)

    print('Done')

    all_errors = np.asarray(all_errors)

    # Plot Data Histograms

    f, ax_arr = plt.subplots(3, 3)

    xyzlwh_bins = 51
    ry_bins = 31
    iou_bins = 51

    # xyz
    ax_arr[0, 0].hist(all_errors[:, 0], xyzlwh_bins, facecolor='green', alpha=0.75)
    ax_arr[0, 1].hist(all_errors[:, 1], xyzlwh_bins, facecolor='green', alpha=0.75)
    ax_arr[0, 2].hist(all_errors[:, 2], xyzlwh_bins, facecolor='green', alpha=0.75)

    # lwh
    ax_arr[1, 0].hist(all_errors[:, 3], xyzlwh_bins, facecolor='green', alpha=0.75)
    ax_arr[1, 1].hist(all_errors[:, 4], xyzlwh_bins, facecolor='green', alpha=0.75)
    ax_arr[1, 2].hist(all_errors[:, 5], xyzlwh_bins, facecolor='green', alpha=0.75)

    # orientation
    ax_arr[2, 0].hist(all_errors[:, 6], ry_bins, facecolor='green', alpha=0.75)

    # iou
    ax_arr[2, 2].hist(all_3d_ious, iou_bins, facecolor='green', alpha=0.75)

    plt.show()


if __name__ == "__main__":
    main()
