import tensorflow as tf
from wavedata.tools.core.depth_map_utils import project_depths
from fcd.utils.utils import get_lidar_point_cloud, read_calibration
import cv2
import matplotlib.pyplot as plt
import numpy as np
from occlusion_mask_layer import OccMaskLayer

img_ph = tf.placeholder(tf.float32, (None,None,None,3), 'img_input')

depth_ph = tf.placeholder(tf.float32, (None,None), 'depth_input')

dataset = 'training'
method = 'sample_dist'
seq_idx = 17
img_idx = 7
velo_dir = '/media/j7deng/Data/track/Kitti-dataset/tracking/data_tracking_velodyne/'+dataset+'/velodyne'
calib_dir= '/media/j7deng/Data/track/Kitti-dataset/tracking/data_tracking_calib/'+dataset+'/calib'
img_dir = '/media/j7deng/Data/track/Kitti-dataset/tracking/data_tracking_image_2/' +dataset+'/image_02'

img_dir_seq = img_dir + "/%04d/%06d.png"  %(seq_idx,img_idx)
velo_dir_seq = velo_dir + "/%04d" %seq_idx

frame_calib = read_calibration(calib_dir=calib_dir, seq_idx=seq_idx)
#velo_dir_seq = velo_dir + "/%04d" %seq_idx
img0 = cv2.imread(img_dir_seq)
image_shape = [img0.shape[1],img0.shape[0]]
point_cloud = get_lidar_point_cloud(img_idx=img_idx, frame_calib=frame_calib, velo_dir=velo_dir_seq,im_size=image_shape)
depth_map = project_depths(point_cloud, frame_calib.p2, [img0.shape[0],img0.shape[1]])

x1f,y1f = 559, 141
x2f,y2f = 627, 350
x1e, y1e,x2e,y2e = 520, 136, 536, 193
boxes = tf.Variable([[y1f,x1f,y2f,x2f],[y1e,x1e,y2e,x2e]])
boxes_norm = tf.Variable([[y1f/370.0,x1f/1224.0,y2f/370.0,x2f/1224.0],[y1e/370.0,x1e/1224.0,y2e/370.0,x2e/1224.0]])
ref_depth_min = tf.Variable([5.3,24])
ref_depth_max = tf.Variable([7.3,26])
n_split = 4
img_size = (32,32)
occ_mak_layer = OccMaskLayer()
box_indices = tf.zeros([2],dtype = tf.int32)
depth_input = tf.expand_dims(depth_ph,0)
depth_input = tf.expand_dims(depth_input,-1)
occ_mask = occ_mak_layer.build(depth_input,boxes_norm,box_indices,ref_depth_min,ref_depth_max,n_split,[16,16],img_size)
crop_img = tf.image.crop_and_resize(img_ph/255,boxes_norm,tf.zeros([2],dtype=tf.int32),img_size)
#mask_tile = tf.tile(occ_mask,[],-1)
masked_img = tf.multiply(crop_img,occ_mask,name='masked_img')

init_op = tf.global_variables_initializer()
#sess= tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess = tf.Session()
sess.run(init_op)
#print(sess.run(occ_mask,feed_dict={img_ph:[img0],depth_ph:depth_map}))
sub_img_masked = sess.run(masked_img,feed_dict={img_ph:[img0],depth_ph:depth_map})
sub_img = sess.run(crop_img,feed_dict={img_ph:[img0],depth_ph:depth_map})

plt.subplot(221)
plt.imshow(sub_img[0,:,:,:])

plt.subplot(222)
plt.imshow(sub_img[1,:,:,:])

plt.subplot(223)
plt.imshow(sub_img_masked[0,:,:,:])

plt.subplot(224)
plt.imshow(sub_img_masked[1,:,:,:])

plt.show()
