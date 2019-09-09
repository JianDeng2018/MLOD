import os
import unittest
import numpy as np
import time
import cv2

from wavedata.tools.obj_detection import obj_utils as od
from wavedata.tools.obj_detection import data_aug as augment

ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))


class AugmentationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_dir = ROOTDIR + "/test_data/"
        cls.test_data_labels = ROOTDIR + "/test_data/label/"

    def test_2d_image_label_flip(self):
        # this is a dummy image, we just
        # want to test the label flipping
        image_sampl = self.data_dir + '/image/000194.png'
        dummy_img = cv2.imread(image_sampl)

        ##########
        # Case 1
        ##########
        label = np.array([10, 170, 20, 296])
        _, flipped_label = augment.flip_2d_image_and_label(dummy_img,
                                                           label)
        # half-width = 621.0
        # calculate by hand where the flipped x's should lie
        expected_label = np.array([1222, 170, 1232, 296])
        np.testing.assert_array_equal(flipped_label,
                                      expected_label)

        ##########
        # Case 2
        ##########
        label = np.array([630, 170, 700, 296])
        _, flipped_label = augment.flip_2d_image_and_label(dummy_img,
                                                           label)

        expected_label = np.array([542, 170, 612, 296])
        np.testing.assert_array_equal(flipped_label,
                                      expected_label)

        ##########
        # Case 3
        ##########
        label = np.array([621, 170, 700, 296])
        _, flipped_label = augment.flip_2d_image_and_label(dummy_img,
                                                           label)

        expected_label = np.array([542, 170, 621, 296])
        np.testing.assert_array_equal(flipped_label,
                                      expected_label)

        ##########
        # Case 4
        ##########
        label = np.array([500, 170, 621, 296])
        _, flipped_label = augment.flip_2d_image_and_label(dummy_img,
                                                           label)

        expected_label = np.array([621, 170, 742, 296])
        np.testing.assert_array_equal(flipped_label,
                                      expected_label)

    def test_3d_pc_label_flip(self):
        objects = od.read_labels(self.test_data_labels, 5258)

        flipped_obj = augment.flip_3d_point_and_label(objects[0])
        # ry should be flipped
        expected_ry = 5.0815926
        self.assertAlmostEqual(flipped_obj.ry,
                               expected_ry,
                               places=5)
        # only t.x should be flipped
        self.assertEqual(objects[0].t[0], -flipped_obj.t[0])
        self.assertEqual(objects[0].t[1], flipped_obj.t[1])
        self.assertEqual(objects[0].t[2], flipped_obj.t[2])

    def test_negative_2d_bb_speed(self):

        object_labels = od.read_labels(self.test_data_labels, 5258)

        first_obj = object_labels[0]
        img_class = "All"
        boxes2d, _, _ = od.build_bbs_from_objects(object_labels,
                                                  img_class)
        iou_threshold_min = 0.01
        iou_threshold_max = 0.7
        samples = 5000

        self.startTime = time.time()
        new_boxes, failed_count = augment.calculate_negative_2d_bb(
                                    first_obj,
                                    boxes2d,
                                    iou_threshold_min,
                                    iou_threshold_max,
                                    samples,
                                    rand_sampl=True)

        t = time.time() - self.startTime
        print('\n==== Random Sampling ====\n')
        print("%s Run Time: \n%.3f" % (self.id(), t))
        print('Number of failures ', failed_count)

        self.startTime = time.time()
        new_boxes, failed_count = augment.calculate_negative_2d_bb(
                                    first_obj,
                                    boxes2d,
                                    iou_threshold_min,
                                    iou_threshold_max,
                                    samples,
                                    rand_sampl=False)

        t = time.time() - self.startTime
        print("\n==== Calculating and Random Sampling on Failures ====\n")
        print("%s Run Time: \n%.3f" % (self.id(), t))
        print('Number of failures ', failed_count)


if __name__ == '__main__':
    unittest.main()
