import unittest
import numpy as np

from wavedata.tools.core.integral_image import IntegralImage


class EvaluationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        test_mat = np.ones((3, 3, 3)).astype(np.float32)
        cls.test_image = test_mat

    def test_integral_image(self):

        # Generate integral image
        integral_image = IntegralImage(self.test_image)
        cuboid = np.array([[0, 0, 0, 1, 1, 1],
                           [0, 0, 0, 2, 2, 2],
                           [0, 0, 0, 3, 3, 3]]).T.astype(np.uint32)

        occupancy_count = integral_image.query(cuboid)

        # First cuboid case = should be 1*1*1 = 1
        self.assertTrue(occupancy_count[0] == 1)
        # Second cuboid case = should be 2*2*2 = 8
        self.assertTrue(occupancy_count[1] == 8)
        # Third cuboid case = should be 3*3*3 = 27
        self.assertTrue(occupancy_count[2] == 27)

        cuboid = np.array([[1, 1, 1, 2, 2, 2],
                           [1, 1, 1, 3, 3, 3]]).T.astype(np.uint32)

        occupancy_count = integral_image.query(cuboid)

        # First cuboid case = should be 1*1*1 = 1
        self.assertTrue(occupancy_count[0] == 1)

        # Second cuboid case = should be 2*2*2 = 8
        self.assertTrue(occupancy_count[1] == 8)

        cuboid = np.array([[0, 0, 0, 3, 3, 1]]).T.astype(np.uint32)
        occupancy_count = integral_image.query(cuboid)

        # Flat Surface case = should be 1*9 = 9
        self.assertTrue(occupancy_count[0] == 9)

        # Test outside the boundary
        cuboid = np.array([[0, 0, 0, 12421, 2312, 162]]).T.astype(np.uint32)
        occupancy_count = integral_image.query(cuboid)
        self.assertTrue(occupancy_count[0] == 27)


if __name__ == '__main__':
    unittest.main()
