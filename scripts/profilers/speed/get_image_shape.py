import cv2
import numpy as np
from PIL import Image

from mlod.builders.dataset_builder import DatasetBuilder

@profile
def main():

    dataset = DatasetBuilder.build_kitti_dataset(DatasetBuilder.KITTI_TRAIN)

    image_path = dataset.get_rgb_image_path('000000')

    # cv2
    cv2_image1 = cv2.imread(image_path)
    cv2_image2 = cv2.imread(image_path)
    cv2_image3 = cv2.imread(image_path)

    cv2_image1_shape = cv2_image1.shape
    cv2_image2_shape = cv2_image2.shape
    cv2_image3_shape = cv2_image3.shape

    print(cv2_image1_shape)
    print(cv2_image2_shape)
    print(cv2_image3_shape)

    # PIL
    pil_image1 = Image.open(image_path)
    pil_image2 = Image.open(image_path)
    pil_image3 = Image.open(image_path)

    pil_image1_shape = pil_image1.size
    pil_image2_shape = pil_image2.size
    pil_image3_shape = pil_image3.size

    print(pil_image1_shape)
    print(pil_image2_shape)
    print(pil_image3_shape)

    pil_image1 = np.asarray(pil_image1)
    pil_image2 = np.asarray(pil_image2)
    pil_image3 = np.asarray(pil_image3)


if __name__ == '__main__':
    main()
