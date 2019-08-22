import csv
import numpy as np
import time
import pandas

import mlod


@profile
def main():

    # Profile large csv
    csv_file_path = mlod.root_dir() + \
        '/data/mini_batches/iou_2d/kitti/train/depth/People[ 0.3  0.3]/007463.csv'

    ###################
    # Numpy loadtxt
    ###################
    np_data = np.loadtxt(csv_file_path, delimiter=',')

    ###################
    # Numpy genfromtxt
    ###################
    np_data = np.genfromtxt(csv_file_path, delimiter=',')

    ###################
    # Pandas read_csv
    ###################
    start_time = time.time()
    pandas_data = pandas.read_csv(csv_file_path).values
    print('pandas.read_csv took :', time.time() - start_time)
    start_time = time.time()

    # Profile small csv
    calib_dir = mlod.root_dir() + \
        '/tests/datasets/Kitti/object/training/calib/000001.txt'

    start_time = time.time()
    data_file = open(calib_dir, 'r')
    csv_reader = csv.reader(data_file, delimiter=' ')
    data = []
    for row in csv_reader:
        data.append(row)
    print('csv.reader took :', time.time() - start_time)
    pandas_data = pandas.read_csv(csv_file_path).values


if __name__ == '__main__':
    main()
