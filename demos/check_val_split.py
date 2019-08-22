import numpy as np

base_path = '/home/j7deng/Kitti/object/'

train_dir = base_path + 'train.txt'
val_dir = base_path + 'val.txt'

with open(train_dir) as f:
    train_sample_name = f.readlines()
    #print(sample_name)

with open(val_dir) as f:
    val_sample_name = f.readlines()

for val_sample in val_sample_name:
    print(val_sample)
    if val_sample in train_sample_name:
        print(val_sample, 'in')
