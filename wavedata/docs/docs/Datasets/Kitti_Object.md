# KITTI Object Detection Benchmark Documentation

This documentation's purpose is the following:

* Provide directory level organization structure of the dataset.
* Provide explanation of the content of files (i.e. calibration params, image format)

The current plan for the dataset organization structure is heavily based on KITTI Vision Benchmark Suite [1].

## Dataset Directory Organization
The following directory is an example from KITTI Dataset structure.
```
                        .(Kitti)
                        └── object
                            ├── testing
                            │   ├── calib
                            │   ├── image_2
                            │   └── velodyne
                            ├── test.txt
                            ├── training
                            │   ├── calib
                            │   ├── image_2
                            │   ├── label_2
                            │   └── velodyne
                            ├── train.txt
                            ├── trainval.txt
                            └── val.txt
```

The object directory contains all the components that are required for object detection on the dataset.
The object directory has 2 main folders:

The data is divided to 7481 training and 7518 test images for the object detection task. 

Inside the training and the testing directory, 3 important folders are listed as follows:

  - **image_02**: contains the left color camera images
  - **label_02**: contains the left color camera label files
  - **calib**: contains the calibration for all four cameras
  - **velodyne**: binary files containing velodyne pointcloud
    
The number of the directory specifies which camera image data it contains. (i.e. 02 is the center color camera) The 
number that precedes the file extension describes the image sequence number which correspends with the calibration file.
  
### Calibration File Format

Image taken from the following paper [1].
![setup](/images/setup_top_view.png)

The calibration is a txt file associated with the same sequence number for the given images for detection.
There are following parameters for each calibration files.

- **P0**: Intrinsic Camera Calibration for Camera 0
- **P1**: Intrinsic Camera Calibration for Camera 1
- **P2**: Intrinsic Camera Calibration for Camera 2
- **P3**: Intrinsic Camera Calibration for Camera 3
- **R0_rect**: Rectification matrix for stereo setup
- **Tr_velo_to_cam**: Extrinsic Transformation from Velodyne to Camera 0
- **Tr_imu_to_velo**: Extrinsic Tranfromation from IMU to Velodyne

### Label Object Documentation 

| Values | Name | Description |
| :-------------: |:-------------:| :--------|
| 1 | `type` | Describes the type of object: 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc' or 'DontCare'|
| 1 | `truncated` | Float from 0 (non-truncated) to 1 (truncated), where truncated refers to the object leaving image boundaries |
| 1 | `occluded` | Integer (0,1,2,3) indicating occlusion state: 0 = fully visible, 1 = partly occluded, 2 = largely occluded, 3 = unknown |
| 1 | `alpha` | Observation angle of object, ranging [-pi..pi] |
| 4 | `bbox` | 2D bounding box of object in the image (0-based index): contains **left, top, right, bottom** pixel coordinates
| 3 | `dimensions` | 3D object dimensions: height, width, length (in meters) |
| 3 | `location` | 3D object location x,y,z in camera coordinates (in meters) |
| 1 | `rotation_y` | Rotation ry around Y-axis in camera coordinates [-pi..pi] |
| 1 | `score` | Only for results: Float, indicating confidence in detection, needed for p/r curves, higher is better.


## Citations

[1] Geiger, A., Lenz, P., Urtasun, R.: Are we ready for autonomous
driving?, The KITTI vision benchmark suite. In: IEEE Computer
Society Conference on Computer Vision and Pattern Recognition
(CVPR), pp. 3354–3361. (2012)
