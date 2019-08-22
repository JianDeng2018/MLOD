import cv2
import numpy as np

from PIL import Image

import mlod

@profile
def main():

    img_file_path = mlod.root_dir() + \
                     '/tests/datasets/Kitti/object/training/image_2/000000.png'

    pil_rgb_image = np.asarray(Image.open(img_file_path))

    cv_bgr_image = cv2.imread(img_file_path)

    cv_rgb_image = cv_bgr_image[..., :: -1]

    cv_cvt_color = cv2.cvtColor(cv_bgr_image, cv2.COLOR_BGR2RGB)
    cv_cvt_color = cv2.cvtColor(cv_bgr_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(pil_rgb_image, (1590, 480))

if __name__ == '__main__':
    main()
