import os
import unittest
import numpy as np
from wavedata.tools.obj_detection import evaluation
from wavedata.tools.obj_detection import obj_utils as utils

ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))


class EvaluationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.gtbb_2d = np.array([50, 50, 100, 100])
        cls.test_cases_2d = np.array([[50, 50, 100, 100],  # test for same
                                      # test for normal intersection
                                      [75, 75, 125, 125],
                                      # test for intersection at single vertex
                                      [0, 0, 50, 50],
                                      # test for intersection at single vertex
                                      [100, 100, 150, 150],
                                      [0, 0, 30, 30],  # test with miss
                                      [150, 150, 210, 210],  # test with miss
                                      # test with container
                                      [25, 25, 100, 100],
                                      # test with container
                                      [25.5, 25.5, 125.5, 125.5]
                                      ])

        cls.gt_iou_2d = np.array([1.0,
                                  (25 * 25) / (2 * 50 * 50 - 25 * 25),
                                  0.0,
                                  0.0,
                                  0.0,
                                  0.0,
                                  (50 * 50) / (75 * 75),
                                  (50 * 50) / (100 * 100)])

        cls.gtbb_3d = np.array([0, 5, 10, 20, 0, 0, 0])

        cls.test_cases_3d = np.array([[0.0, 5, 10, 20, 0, 0, 0],  # test for same
                                      # test for normal intersection
                                      [0.0, 5, 10, 20, 0, 5, 0],
                                      # test with miss
                                      [0.0, 5, 10, 20, 30, 30, 30],
                                      # test with rotated
                                      [np.pi / 3, 50, 10, 100, 0, 0, 0],
                                      [np.pi / 3, 5, 10, 20, 0, 0, 0]])

        cls.rotated_intersection_gt = np.array([100.0,
                                                100.0,
                                                0,
                                                100.0,
                                                28.975])
        cls.gt_iou_3d = np.array([1.0,
                                  0.3333,
                                  0.0,
                                  0.0200,
                                  0.16941967548604006])

        cls.gt_h_int = np.array([10, 5, 0])
        cls.gt_h_un = np.array([10, 15, 20])

        cls.label_dir = ROOTDIR + '/test_data/label'
        cls.results_dir = ROOTDIR + '/test_data/detections'

    def test_2d_iou(self):
        iou = evaluation.two_d_iou(self.gtbb_2d, self.test_cases_2d)
        np.testing.assert_allclose(self.gt_iou_2d, iou, 0, 0.01)

        gt_box = np.asarray([0.0, 0.0, 0.5, 0.5])
        test_boxes = np.asarray(
            [
                [-0.5, 0.0, 0.0, 0.5],
                [-0.25, 0.0, 0.25, 0.5],
                [0.0, 0.0, 0.5, 0.5],
                [0.25, 0.0, 0.75, 0.5],
                [0.5, 0.0, 1.0, 0.5],
            ])
        exp_ious_2d = [0.0, 0.333, 1.0, 0.333, 0.0]
        ious_2d = evaluation.two_d_iou(gt_box, test_boxes)
        np.testing.assert_almost_equal(ious_2d, exp_ious_2d)

    def test_height_metrics(self):
        h_int, h_un = evaluation.height_metrics(self.gtbb_3d,
                                                self.test_cases_3d[0:3])
        np.testing.assert_almost_equal(self.gt_h_int, h_int)
        np.testing.assert_almost_equal(self.gt_h_un, h_un)

    def test_get_rotated_3d_bb(self):
        x, z = evaluation.get_rotated_3d_bb(self.test_cases_3d[3:4])
        x_true = np.array([55.8013, -30.8013, -55.8013, 30.8013])
        z_true = np.array([3.3494, -46.6506, -3.3494, 46.6506])

        np.testing.assert_almost_equal(x_true, x, 3)
        np.testing.assert_almost_equal(z_true, z, 3)

        x, z = evaluation.get_rotated_3d_bb(self.test_cases_3d[2:4])
        x_true = np.array([[32.5, 32.5, 27.5, 27.5],
                           [55.8013, -30.8013, -55.8013, 30.8013]])
        z_true = np.array([[40.0, 20, 20, 40],
                           [3.3494, -46.6506, -3.3494, 46.6506]])
        np.testing.assert_almost_equal(x_true, x, 3)
        np.testing.assert_almost_equal(z_true, z, 3)

    def test_get_rectangular_metrics(self):
        rect_met = evaluation.get_rectangular_metrics(self.gtbb_3d,
                                                      self.test_cases_3d)
        np.testing.assert_almost_equal(self.rotated_intersection_gt,
                                       rect_met, 3)

        rect_met = evaluation.get_rectangular_metrics(self.gtbb_3d,
                                                      self.test_cases_3d[4])
        np.testing.assert_almost_equal(self.rotated_intersection_gt[4],
                                       rect_met, 3)

    def test_3d_iou(self):
        iou = evaluation.three_d_iou(self.gtbb_3d, self.test_cases_3d)
        np.testing.assert_almost_equal(self.gt_iou_3d,
                                       iou, 3)

        iou = evaluation.three_d_iou(self.gtbb_3d, self.test_cases_3d[4])
        np.testing.assert_almost_equal(self.gt_iou_3d[4],
                                       iou, 3)

    def test_evaluate_2d(self):
        gt_detections = utils.read_labels(self.label_dir, 194)
        detections = utils.read_labels(self.results_dir, 194, True)
        gt_2d_bbs, _, _ = utils.build_bbs_from_objects(gt_detections,
                                                       ['Car', 'car'])

        detected_2d_bbs, _, scores = utils.build_bbs_from_objects(detections, [
                                                                  'Car', 'car'])

        score_thresholds = np.array([0.5])
        iou_threshold = 0.1

        car_precision_2d, car_recall_2d = evaluation.evaluate_2d(
            [gt_2d_bbs],
            [detected_2d_bbs],
            [scores],
            score_thresholds,
            iou_threshold)
        np.testing.assert_almost_equal(car_precision_2d,
                                       1.0, 3)

        np.testing.assert_almost_equal(car_recall_2d,
                                       1.0, 3)

        score_thresholds = np.array([0.994])
        car_precision_2d, car_recall_2d = evaluation.evaluate_2d(
            [gt_2d_bbs],
            [detected_2d_bbs],
            [scores],
            score_thresholds,
            iou_threshold)
        np.testing.assert_almost_equal(car_precision_2d,
                                       1.0, 3)

        np.testing.assert_almost_equal(car_recall_2d,
                                       0.5, 3)

    def test_evaluate_3d(self):
        gt_detections = utils.read_labels(self.label_dir, 194)
        detections = utils.read_labels(self.results_dir, 194, True)
        _, gt_3d_bbs, _ = utils.build_bbs_from_objects(gt_detections, ['Car',
                                                                       'car'])

        _, detected_3d_bbs, scores = utils.build_bbs_from_objects(detections, [
                                                                  'Car', 'car'])

        score_thresholds = np.array([0.5])
        iou_threshold = 0.25

        car_precision_3d, car_recall_3d = evaluation.evaluate_3d(
            [gt_3d_bbs],
            [detected_3d_bbs],
            [scores],
            score_thresholds,
            iou_threshold)

        np.testing.assert_almost_equal(car_precision_3d,
                                       1.0, 3)

        np.testing.assert_almost_equal(car_recall_3d,
                                       1.0, 3)

        score_thresholds = np.array([0.994])
        car_precision_3d, car_recall_3d = evaluation.evaluate_3d(
            [gt_3d_bbs],
            [detected_3d_bbs],
            [scores],
            score_thresholds,
            iou_threshold)

        np.testing.assert_almost_equal(car_precision_3d,
                                       1.0, 3)

        np.testing.assert_almost_equal(car_recall_3d,
                                       0.5, 3)


if __name__ == '__main__':
    unittest.main()
