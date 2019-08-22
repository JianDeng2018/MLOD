import os

import cv2
import numpy as np

from wavedata.tools.obj_detection import data_aug
from wavedata.tools.visualization import vis_utils


def main():

    kitti_dir = os.path.expanduser('~/Kitti/object/training/image_2')

    sample_name = '000032'
    # sample_name = '000041'
    # sample_name = '000991'
    # sample_name = '006338'

    image_bgr = cv2.imread('{}/{}.png'.format(kitti_dir, sample_name))

    # PCA Jitter
    pca = data_aug.compute_pca([image_bgr])
    image_pca = data_aug.add_pca_jitter(image_bgr, pca)

    # Swap B and G channels in RGB
    image_bg_swap = np.copy(image_bgr)
    image_bg_swap[:, :, 0], image_bg_swap[:, :, 1] = \
        image_bg_swap[:, :, 1], image_bg_swap[:, :, 0]

    # Gaussian noise
    gaussian_noise = np.random.randn(*image_bgr.shape) * 20.0
    image_gaussian = np.uint8(
        np.clip(image_bgr + gaussian_noise, 0.0, 255.0))

    # Random noise
    random_noise = np.random.rand(*image_bgr.shape) * 40.0 - 20.0
    image_random = np.uint8(
        np.clip(image_bgr + random_noise, 0.0, 255.0))

    # Channel specific noise
    channel_gaussian_noise = np.random.randn(3) * 10.0
    image_channel_gaussian = np.uint8(
        np.clip(image_bgr + channel_gaussian_noise, 0.0, 255.0))

    # Brightness
    random_brightness = np.random.randn(1) * 20.0
    image_brightness = np.uint8(
        np.clip(image_bgr + random_brightness, 0.0, 255.0))

    img_size = (620, 220)

    vis_utils.cv2_show_image('BGR', image_bgr,
                             img_size, (0, 0))
    vis_utils.cv2_show_image('image_pca', image_pca,
                             img_size, (620, 0))
    vis_utils.cv2_show_image('image_bg_swap', image_bg_swap,
                             img_size, (1240, 0))
    vis_utils.cv2_show_image('image_gaussian', image_gaussian,
                             img_size, (0, 220))
    vis_utils.cv2_show_image('image_random', image_random,
                             img_size, (620, 220))
    vis_utils.cv2_show_image('image_channel_gaussian', image_channel_gaussian,
                             img_size, (1200, 220))
    vis_utils.cv2_show_image('image_brightness', image_brightness,
                             img_size, (0, 440))

    cv2.waitKey()


if __name__ == '__main__':
    main()

