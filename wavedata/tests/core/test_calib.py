import os
import unittest
import numpy as np
import scipy.io
#import cv2
from wavedata.tools.core import calib_utils as calib

ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


class EvaluationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        test_data_dir = ROOTDIR + "/tests/test_data/"
        cam_0 = scipy.io.loadmat(test_data_dir+'camera_0.mat')
        cam_1 = scipy.io.loadmat(test_data_dir+'camera_1.mat')
        cls.stereo = scipy.io.loadmat(test_data_dir+'stereo.mat')
        cls.test_cases = [cam_0, cam_1]
        cls.point_cloud = scipy.io.loadmat(test_data_dir+'pcl.mat')

    def test_krt_from_p(self):
        for i in self.test_cases:
            p = i['P']
            k_true = i['Kl']
            r_true = i['Rotl']
            t_true = i['tl']
            t_true = np.transpose(t_true)
            t_true = t_true[0]
            k_test, r_test, t_test = calib.krt_from_p(p)
            np.testing.assert_almost_equal(k_test, k_true, 4)
            np.testing.assert_almost_equal(r_test, r_true, 4)
            np.testing.assert_almost_equal(t_test,
                                           t_true,
                                           4)

    def test_stereo_calibration(self):
        cam_0 = self.test_cases[0]
        cam_1 = self.test_cases[1]

        stereo_info = calib.get_stereo_calibration(cam_0['P'],
                                                   cam_1['P'])

        self.assertAlmostEqual(stereo_info.baseline,
                               self.stereo['baseline'])

        self.assertAlmostEqual(stereo_info.f,
                               self.stereo['f'])

        self.assertAlmostEqual(stereo_info.center_u,
                               self.stereo['cu'])

        self.assertAlmostEqual(stereo_info.center_v,
                               self.stereo['cv'])

    def test_depth_from_disparity(self):
        # Just to check if method works without errors.
        calib_dir = ROOTDIR + '/tests/test_data/calib'
        disp_dir = ROOTDIR + '/tests/test_data'
        img_idx = 1
        disp = calib.read_disparity(disp_dir, img_idx)
        frame_calib = calib.read_calibration(calib_dir, img_idx)
        stereo_calibration_info = calib.get_stereo_calibration(frame_calib.p2,
                                                               frame_calib.p3)

        x, y, z = calib.depth_from_disparity(disp, stereo_calibration_info)

    def test_read_calibration(self):
        # The test file used for calibration is 724513.txt
        test_data_dir = ROOTDIR + "/tests/test_data/calib/"
        calib_out = calib.read_calibration(test_data_dir, 724513)
        test_data_dir = ROOTDIR + "/tests/test_data/"
        calib_true = scipy.io.loadmat(test_data_dir+'readcalib.mat')

        np.testing.assert_almost_equal(calib_out.p0, calib_true['p0'])
        np.testing.assert_almost_equal(calib_out.p1, calib_true['p1'])
        np.testing.assert_almost_equal(calib_out.p2, calib_true['p2'])
        np.testing.assert_almost_equal(calib_out.p3, calib_true['p3'])
        np.testing.assert_almost_equal(calib_out.r0_rect, calib_true['r0_rect'])

        np.testing.assert_almost_equal(calib_out.tr_velodyne_to_cam,
                                       calib_true['tr_velo_to_cam'])

    @unittest.skip
    def test_read_lidar(self):
        test_data_dir = ROOTDIR + "/tests/test_data/calib"
        velo_mat = scipy.io.loadmat(test_data_dir + '/test_velo.mat')
        velo_true = velo_mat['current_frame']['xyz_velodyne'][0][0][:,0:3]

        x, y, z, i = calib.read_lidar(velo_dir=test_data_dir,
                                      img_idx=0)

        velo_test = np.vstack((x, y, z)).T
        np.testing.assert_almost_equal(velo_true, velo_test, decimal=5, verbose=True)

        velo_mat = scipy.io.loadmat(test_data_dir + '/test_velo_tf.mat')
        velo_true_tf = velo_mat['velo_cam_frame']

        calib_out = calib.read_calibration(test_data_dir, 0)
        xyz_cam = calib.lidar_to_cam_frame(velo_test, calib_out)

        np.testing.assert_almost_equal(velo_true_tf, xyz_cam, decimal=5, verbose=True)


if __name__ == '__main__':
    unittest.main()

