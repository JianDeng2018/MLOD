#!usr/bin/env python

import unittest
import numpy as np
import os
from wavedata.tools.core import xyz_enc_utils

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


class XyzEncTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.point_cloud = np.array([[-0.4, 0.5, 0.5],  # original point
                                    [1.3, 5.2, 4.4],  # random second point
                                    [-0.4, 0.5, 0.5],  # original point again
                                    [-0.4, 0.5, 50],  # same projection as original point, but larger depth
                                    [-0.4, 0.5, 0.3],  # same projection as original point, but shorter depth
                                    ])
        cls.image_shape = [2, 3]
        cls.point_in_im_rounded = np.array([[0, 1],
                                            [2, 0],
                                            [0, 1],
                                            [0, 1],
                                            [0, 1]])

    def testCollisionCheck(self):
        # case 1: same point ---> take last point iterated
        same_point_pcl = self.point_cloud[0:3][:].T
        same_point_in_im_rounded = self.point_in_im_rounded[0:3][:]
        same_ind = xyz_enc_utils.collision_check(same_point_pcl,
                                                 self.image_shape,
                                                 len(same_point_pcl.T),
                                                 same_point_in_im_rounded)
        correct_out = np.array([[np.inf, np.inf, 1.0],
                                [2.0, np.inf, np.inf]])
        np.testing.assert_almost_equal(same_ind, correct_out)

        # case 2: distance collision check ---> take smallest distance to cam
        all_ind = xyz_enc_utils.collision_check(self.point_cloud.T,
                                                self.image_shape,
                                                len(self.point_cloud),
                                                self.point_in_im_rounded)
        correct_ind = np.array([[np.inf, np.inf, 1.0],
                                [4.0, np.inf, np.inf]])
        np.testing.assert_almost_equal(all_ind, correct_ind)

    def testFillXyz(self):
        all_x, all_y, all_z = xyz_enc_utils.fill_xyz(self.point_cloud.T,
                                                     self.image_shape,
                                                     len(self.point_cloud),
                                                     self.point_in_im_rounded)

        all_x_correct = np.array([[np.NaN, np.NaN, 1.3],
                                  [-0.4, np.NaN, np.NaN]])
        all_y_correct = np.array([[np.NaN, np.NaN, 5.2],
                                  [0.5, np.NaN, np.NaN]])
        all_z_correct = np.array([[np.NaN, np.NaN, 4.4],
                                  [0.3, np.NaN, np.NaN]])

        np.testing.assert_almost_equal(all_x, all_x_correct)
        np.testing.assert_almost_equal(all_y, all_y_correct)
        np.testing.assert_almost_equal(all_z, all_z_correct)


if __name__ == '__main__':
    unittest.main()
