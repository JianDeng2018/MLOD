import os
import unittest
import scipy.io
import numpy as np
from wavedata.tools.obj_detection import obj_utils
from wavedata.tools.core import calib_utils as calib

ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


class EvaluationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_data_dir = ROOTDIR + "/tests/test_data"
        cls.test_data_label_dir = cls.test_data_dir + "/label/"
        cls.test_data_calib_dir = cls.test_data_dir + "/calib/"
        cls.test_data_planes_dir = cls.test_data_dir + "/planes/"

    def test_read_labels(self):
        # The test file used for calibration is 005258.txt
        label_out = obj_utils.read_labels(self.test_data_label_dir, 5258)
        label_true = scipy.io.loadmat(self.test_data_dir + '/readlabel.mat')

        # Check Size
        self.assertEqual(len(label_out), label_true['label'][0].shape[0])

        # Check if the labels are correct from the test data
        for i in range(0, len(label_out)):
            self.assertTrue(label_out[i].type == label_true['label'][0][i][0])
            self.assertTrue(label_out[i].truncation ==
                            label_true['label'][0][i][1])
            self.assertTrue(label_out[i].occlusion ==
                            label_true['label'][0][i][2])
            self.assertTrue(label_out[i].alpha == label_true['label'][0][i][3])
            self.assertTrue(label_out[i].x1 == label_true['label'][0][i][4])
            self.assertTrue(label_out[i].y1 == label_true['label'][0][i][5])
            self.assertTrue(label_out[i].x2 == label_true['label'][0][i][6])
            self.assertTrue(label_out[i].y2 == label_true['label'][0][i][7])
            self.assertTrue(label_out[i].h == label_true['label'][0][i][8])
            self.assertTrue(label_out[i].w == label_true['label'][0][i][9])
            self.assertTrue(label_out[i].l == label_true['label'][0][i][10])
            self.assertTrue((label_out[i].t ==
                             label_true['label'][0][i][11]).all())
            self.assertTrue(label_out[i].ry == label_true['label'][0][i][12])

    def test_compute_box_3d(self):
        # read in calib file and label file and mat file
        calib_frame = calib.read_calibration(self.test_data_calib_dir, 724513)
        objects = obj_utils.read_labels(self.test_data_label_dir, 5258)
        label_true = scipy.io.loadmat(self.test_data_dir + '/compute3d.mat')

        # compute
        corners_3d = obj_utils.compute_box_corners_3d(objects[0])
        corners, face_idx = obj_utils.project_box3d_to_image(
            corners_3d, calib_frame.p2)
        # compare data
        np.testing.assert_almost_equal(corners, label_true['corners'])

        orientation = obj_utils.compute_orientation_3d(objects[0], calib_frame.p2)

        # -1 for index in python vs matlab
        self.assertTrue((face_idx == label_true['face_idx']-1).all())

        # Test orientation
        self.assertAlmostEqual(orientation.all(),
                               label_true['orientation'].all())

        return

    def test_get_road_plane(self):
        plane = obj_utils.get_road_plane(0, self.test_data_planes_dir)

        np.testing.assert_allclose(plane, [-7.051729e-03, -9.997791e-01,
                                           -1.980151e-02, 1.680367e+00])

    def test_is_point_inside(self):

        P1 = [1.0, 0.0, 0.0]
        P2 = [0.0, 0.0, 0.0]
        P3 = [0.0, 1.0, 0.0]
        P4 = [1.0, 1.0, 0.0]

        P5 = [1.0, 0.0, 1.0]
        P6 = [0.0, 0.0, 1.0]
        P7 = [0.0, 1.0, 1.0]
        P8 = [1.0, 1.0, 1.0]

        cube_corners = np.zeros((3, 8))
        cube_corners[0][0] = P1[0]
        cube_corners[0][1] = P2[0]
        cube_corners[0][2] = P3[0]
        cube_corners[0][3] = P4[0]
        cube_corners[0][4] = P5[0]
        cube_corners[0][5] = P6[0]
        cube_corners[0][6] = P7[0]
        cube_corners[0][7] = P8[0]

        cube_corners[1][0] = P1[1]
        cube_corners[1][1] = P2[1]
        cube_corners[1][2] = P3[1]
        cube_corners[1][3] = P4[1]
        cube_corners[1][4] = P5[1]
        cube_corners[1][5] = P6[1]
        cube_corners[1][6] = P7[1]
        cube_corners[1][7] = P8[1]

        cube_corners[2][0] = P1[2]
        cube_corners[2][1] = P2[2]
        cube_corners[2][2] = P3[2]
        cube_corners[2][3] = P4[2]
        cube_corners[2][4] = P5[2]
        cube_corners[2][5] = P6[2]
        cube_corners[2][6] = P7[2]
        cube_corners[2][7] = P8[2]

        # This point should lie within the cube
        x = [0.1, 0.2, 0.1]

        point_inside = obj_utils.is_point_inside(x, cube_corners)
        self.assertTrue(point_inside)

        # This should lie outside the cube
        y = [-0.1, 0.0, 0.0]

        point_inside = obj_utils.is_point_inside(y, cube_corners)
        self.assertFalse(point_inside)

    def test_get_point_filter(self):

        xz_plane = [0, -1, 0, 0]

        points = np.array([[0, 1, 0], [0, -1, 0], [5, 1, 5], [-5, 1, 5]])
        point_cloud = points.T

        # Test with offset planes at 0.5, and 2.0 distance
        filter1 = obj_utils.get_point_filter(point_cloud, [[-2, 2], [-2, 2], [-2, 2]],
                                             xz_plane, offset_dist=0.5)
        filter2 = obj_utils.get_point_filter(point_cloud, [[-2, 2], [-2, 2], [-2, 2]],
                                             xz_plane, offset_dist=2.0)

        self.assertEqual(np.sum(filter1), 1)
        self.assertEqual(np.sum(filter2), 2)

        filtered1 = points[filter1]
        filtered2 = points[filter2]

        self.assertEqual(len(filtered1), 1)
        self.assertEqual(len(filtered2), 2)

        np.testing.assert_allclose(filtered1, [[0, 1, 0]])
        np.testing.assert_allclose(filtered2, [[0, 1, 0], [0, -1, 0]])

    def test_object_label_eq(self):
        # Case 1, positive case
        object_1 = obj_utils.ObjectLabel()
        object_2 = obj_utils.ObjectLabel()
        self.assertTrue(object_1 == object_2)

        object_1.t = (1., 2., 3.)
        object_2.t = (1., 2., 3.)
        self.assertTrue(object_1 == object_2)

        # Case 2, negative case (single value)
        object_1 = {}  # Not a object label type
        object_2 = obj_utils.ObjectLabel()
        self.assertFalse(object_1 == object_2)

        object_1 = obj_utils.ObjectLabel()
        object_1.truncation = 1.
        self.assertFalse(object_1 == object_2)

        object_1 = obj_utils.ObjectLabel()
        object_1.occlusion = 1.
        self.assertFalse(object_1 == object_2)

        object_1 = obj_utils.ObjectLabel()
        object_1.alpha = 1.
        self.assertFalse(object_1 == object_2)

        object_1 = obj_utils.ObjectLabel()
        object_1.x1 = 1.
        self.assertFalse(object_1 == object_2)

        object_1 = obj_utils.ObjectLabel()
        object_1.y1 = 1.
        self.assertFalse(object_1 == object_2)

        object_1 = obj_utils.ObjectLabel()
        object_1.x2 = 1.
        self.assertFalse(object_1 == object_2)

        object_1 = obj_utils.ObjectLabel()
        object_1.y2 = 1.
        self.assertFalse(object_1 == object_2)

        object_1 = obj_utils.ObjectLabel()
        object_1.h = 1.
        self.assertFalse(object_1 == object_2)

        object_1 = obj_utils.ObjectLabel()
        object_1.w = 1.
        self.assertFalse(object_1 == object_2)

        object_1 = obj_utils.ObjectLabel()
        object_1.l = 1.
        self.assertFalse(object_1 == object_2)

        object_1 = obj_utils.ObjectLabel()
        object_1.t = (1., 1., 1.)
        self.assertFalse(object_1 == object_2)

        object_1 = obj_utils.ObjectLabel()
        object_1.ry = 1.
        self.assertFalse(object_1 == object_2)

        object_1 = obj_utils.ObjectLabel()
        object_1.score = 1.
        self.assertFalse(object_1 == object_2)

        # Case 2, negative case (multiple values)
        object_1 = obj_utils.ObjectLabel()
        object_1.type = ""  # Type of object
        object_1.truncation = 1.
        object_1.occlusion = 1.
        object_1.alpha = 1.
        object_1.x1 = 1.
        object_1.y1 = 1.
        object_1.x2 = 1.
        object_1.y2 = 1.
        object_1.h = 1.
        object_1.w = 1.
        object_1.l = 1.
        object_1.t = [1., 1., 1.]
        object_1.ry = 1.
        object_1.score = 1.
        self.assertFalse(object_1 == object_2)


if __name__ == '__main__':
    unittest.main()
