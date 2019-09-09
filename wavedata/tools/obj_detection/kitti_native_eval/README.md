# kitti_native_eval

`evaluate_object_3d_offline.cpp`evaluates your KITTI detection locally on your own computer using your validation data selected from KITTI training dataset, with the following metrics:

- Average Precision In 2D Image Frame (AP)
- Oriented overlap on image (AOS)
- Average Precision In BEV (AP)
- Average Precision In 3D (AP)

1. Install dependencies: 
```
sudo apt-get install gnuplot gnuplot5
sudo apt-get install libboost-all-dev
```

2. Compile:
```
cd /kitti_native_eval
make
```

3. Run the evaluation using the following command:
```
./evaluate_object_3d_offline groundtruth_dir result_dir
```

- Place your results in data folder and use /kitti_native_eval as results_dir
- Use ~/Kitti/object/training/label_2  as your groundtruth_dir

Example:
```
./evaluate_object_3d_offline ~/Kitti/object/training/label_2 ./
```

Note that you don't have to detect over all KITTI training data. The evaluator only evaluates samples whose result files exist.


- Results will appear per class in terminal for easy, medium and difficult data.
- Precision-Recall Curves will be generated and saved to 'plot' dir.
- Detections should be in the format:
  - `[type, truncation, occlusion, alpha, (x1, y1, x2, y2), (h, w, l), (x, y, z), ry, score]`
