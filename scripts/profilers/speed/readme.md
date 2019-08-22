# Speed Profiling

## Installation
`pip3 install line_profiler`

## Setup
Just above the function you want to profile, add the
`@profile` decorator. The decorator is only used for profiling with the command `kernprof`, and you will need to remove the decorator to run the script normally.
```
@profile
def get_stereo_point_cloud():
```

---
From the top level mlod folder:
```
kernprof -o OUTFILE.lprof -l FILE.py

e.g.
kernprof -o scripts/profilers/speed/csv_loading.py.lprof -l scripts/profilers/speed/csv_loading.py
```
This outputs an `.lprof` file containing the timing information.

---
Alternatively, you can run the profiler from a specific folder without specifying the OUTFILE:
```
cd scripts/profilers/speed
kernprof -l <file.py>

e.g.
kernprof -l csv_loading.py
```
In this case, a `csv_loading.py.lprof` file will be output to the current directory.

---
## Viewing Data
To view the information:
```
python -m line_profiler <OUTFILE.lprof>
```
If you see `_pickle.UnpicklingError: could not find MARK`, make sure you are running the line_profiler on the `.lprof` file and not the `.py` file.

Example Output:
```
Timer unit: 1e-06 s

Total time: 0.039741 s
File: .../wavedata/tools/obj_detection/obj_utils.py
Function: get_stereo_point_cloud at line 205

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   205                                           @profile
   206                                           def get_stereo_point_cloud(img_idx, calib_dir, disp_dir):
   207                                               """
   208                                               Gets the point cloud for an image calculated from the disparity map
   209
   210                                               :param img_idx: image index
   211                                               :param calib_dir: directory with calibration files
   212                                               :param disp_dir: directory with disparity images
   213
   214                                               :return: point_cloud in the form [[x,...][y,...][z,...]]
   215                                               """
   216
   217         2            7      3.5      0.0      disp = cv2.imread(disp_dir + "/%06d_left_disparity.png" % img_idx,
   218         2        10023   5011.5     25.2                        cv2.IMREAD_ANYDEPTH)
   219
   220                                               # Read calibration info
   221         2          391    195.5      1.0      frame_calib = calib.read_calibration(calib_dir, img_idx)
   222         2            2      1.0      0.0      stereo_calibration_info = calib.get_stereo_calibration(frame_calib.p2,
   223         2         9656   4828.0     24.3                                                             frame_calib.p3)
   224
   225                                               # Calculate the point cloud
   226         2        19659   9829.5     49.5      point_cloud = calib.depth_from_disparity(disp, stereo_calibration_info)
   227
   228         2            3      1.5      0.0      return point_cloud
```
