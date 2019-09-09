import os
import time
import unittest
import cv2
import numpy as np
from wavedata.tools.fs_estimation import evaluation
from demos.cityscape import eval_cityscape
from demos.cityscape.eval_data_cityscape import data_class_8 as data_class

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


class EvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data_dir = os.path.join(ROOT_DIR, "tests/test_data/eval")

    def test_eval_segnet_get_args(self):
        args = eval_cityscape.get_args(
            ["segs_dir", "truths_dir", "-o", "dummy_out"])
        self.assertEqual(args.seg_dir, "segs_dir")
        self.assertEqual(args.truth_dir, "truths_dir")
        self.assertEqual(args.interactive, False)
        self.assertEqual(args.output, "dummy_out")

    def test_eval_segnet_get_frames(self):
        num_frames, _, _ = eval_cityscape.get_frames(
            os.path.join(self.test_data_dir, "SegNet_a"),
            os.path.join(self.test_data_dir, "SegNet_b"))
        self.assertEqual(num_frames, 3)

    def test_eval_segnet_get_input(self):
        eval_cityscape.input = lambda _: "   Dummy_String       "
        eval_cityscape.raw_input = lambda _: "   Dummy_String       "

        input_str = eval_cityscape.get_input(
            "get_input should return lower case, trimmed value")
        self.assertEqual(input_str, "dummy_string")

    def test_eval_segnet_frame_loading(self):
        timestamp = time.time()
        eval_cityscape.load_frameset(
            os.path.join(self.test_data_dir, "SegNet_a/right_size1.png"),
            os.path.join(self.test_data_dir, "SegNet_b/right_size2.png"))
        load_time_ms = 1000 * (time.time() - timestamp)

        self.assertLess(load_time_ms, 130)  # performance test may fail

    def test_eval_segnet_metrics_size_assert(self):
        img1 = cv2.imread(os.path.join(
            self.test_data_dir, "SegNet_b/wrong_size.png"))
        img2 = cv2.imread(os.path.join(
            self.test_data_dir, "SegNet_a/car_square.png"))

        with self.assertRaises(AssertionError):
            evaluation.metrics(img1, img2, data_class)

    def test_eval_segnet_metrics_same(self):
        img1 = cv2.imread(os.path.join(
            self.test_data_dir, "SegNet_a/car_square.png"))
        img2 = cv2.imread(os.path.join(
            self.test_data_dir, "SegNet_a/car_square.png"))

        g, c, miou = evaluation.metrics(img1, img2, data_class)
        self.assertEqual(g, 1)
        np.testing.assert_array_equal(c, np.array([np.nan, np.nan, np.nan,
            np.nan, np.nan, 1.0, np.nan, np.nan]))
        np.testing.assert_array_equal(miou, np.array([np.nan, np.nan, np.nan,
            np.nan, np.nan, 1.0, np.nan, np.nan]))

    def test_eval_segnet_metrics_diff_1(self):
        img1 = cv2.imread(os.path.join(
            self.test_data_dir, "SegNet_a/car_square.png"))
        img2 = cv2.imread(os.path.join(
            self.test_data_dir, "SegNet_b/road_square.png"))

        g, c, miou = evaluation.metrics(img1, img2, data_class)
        self.assertEqual(g, 0.75)
        np.testing.assert_array_equal(c, np.array([0.0, np.nan, np.nan,
            np.nan, np.nan, np.nan, np.nan, np.nan]))
        np.testing.assert_array_equal(miou, np.array([0.0, np.nan, np.nan, np.nan,
            np.nan, 0.0, np.nan, np.nan]))

    def test_eval_segnet_metrics_diff_2(self):
        img1 = cv2.imread(os.path.join(
            self.test_data_dir, "SegNet_a/car_square.png"))
        img2 = cv2.imread(os.path.join(
            self.test_data_dir, "SegNet_a/car_road_square.png"))

        g, c, miou = evaluation.metrics(img1, img2, data_class)
        self.assertEqual(g, 0.75)
        np.testing.assert_array_equal(c, np.array([0.0, np.nan, np.nan,
            np.nan, np.nan, 0.5, np.nan, np.nan]))
        np.testing.assert_array_almost_equal(miou, np.array([0.0, np.nan, np.nan,
            np.nan, np.nan, 0.333333, np.nan, np.nan]))


if __name__ == '__main__':
    unittest.main()
