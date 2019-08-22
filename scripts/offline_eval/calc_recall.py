"""Prepare data, evaluate and plot."""

import os
import numpy as np
import matplotlib.pyplot as plt

from wavedata.tools.obj_detection import evaluation
from wavedata.tools.obj_detection import obj_utils

import mlod
from mlod.core import box_3d_encoder
from mlod.builders.dataset_builder import DatasetBuilder


def main():

    dataset = DatasetBuilder.build_kitti_dataset(
        DatasetBuilder.KITTI_VAL)

    # get proposals
    proposal_output_dir = mlod.root_dir() + \
        "/data/predictions/rpn_model/proposals_and_scores/" + \
         dataset.data_split
    global_steps = os.listdir(proposal_output_dir)
    print('Checkpoints found ', global_steps)

    all_recalls = []

    for step in global_steps:
        for sample_name in dataset.sample_list:
            img_idx = int(sample_name)

            # ------------------------------------
            # Load proposals and scores from files
            # ------------------------------------
            proposals_scores_dir = proposal_output_dir + \
                "{}/{}/{}.txt".format(dataset.data_split,
                                      step,
                                      sample_name)
            if not os.path.exists(proposals_scores_dir):
                print('File {} not found, skipping'.format(sample_name))
                continue
            proposals_scores = np.loadtxt(proposals_scores_dir)

            proposals = proposals_scores[:, 0:-1]
            proposal_iou_format = \
                box_3d_encoder.box_3d_to_3d_iou_format(proposals)
            # scores are in the last column
            scores = proposals_scores[:, -1]

            # -----------------------
            # Get ground truth labels
            # -----------------------
            gt_objects = obj_utils.read_labels(dataset.label_dir, img_idx)
            _, gt_3d_bbs, _ = obj_utils.build_bbs_from_objects(gt_objects,
                                                               ['Car', 'car'])

            score_thresholds = np.array([0.3])
            iou_threshold = 0.0025
            # calculate RPN recall and precision
            precision, recall = evaluation.evaluate_3d(
                                                [gt_3d_bbs],
                                                [proposal_iou_format],
                                                [scores],
                                                score_thresholds,
                                                iou_threshold)

            print('Recall ', recall[0])
            print('Precision ', precision[0])
            all_recalls.append(recall)

    # -------------------------
    # TODO: plot
    # -------------------------
    # flattened_recalls = np.ravel(all_recalls)
    # plt.plot(flattened_recalls)
    # plt.show()


if __name__ == '__main__':
    main()
