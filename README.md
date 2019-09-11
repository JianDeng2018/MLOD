# Multi-view Labelling Object Detector
This repository contains the implementation of our Multi-View 3D Object Detection Based on Robust Feature Fusion Method. This codes are modified from [AVOD](https://github.com/kujason/avod). 

[**MLOD: A multi-view 3D object detection based on robust feature fusion method**](https://arxiv.org/abs/1909.04163)

[Jian Deng](https://scholar.google.com/citations?user=1QvpHZMAAAAJ&hl=en), [Krzysztof Czarnecki](https://scholar.google.ca/citations?user=ZzCpumQAAAAJ&hl=en) 

## Getting Started
Implemented and tested on Ubuntu 16.04, Python 3.5 and Tensorflow 1.9.0.

Add MLOD to PYTHONPATH

``` bash
# From MLOD/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/MLOD
```

or, you can add it to your ~/.bashrc file.

### Protobuf Compilation

Compile the the Protobuf libraries through the following script:

``` bash
# From top level mlod folder
sh mlod/protos/run_protoc.sh
```

Alternatively, you run `protoc` command directly:

``` bash
# From top level mlod folder
protoc mlod/protos/*.proto --python_out=.
```

## Training

### Dataset

To train on [Kitti Dataset](http://www.cvlibs.net/datasets/kitti/), download the data and place it inside home dir:

``` bash
# Kitti Dataset
~/Kitti
```

### Mini-batch Generation

The training data needs to be pre-processed to generate mini-batches. These mini-batches include the anchors the network
trains on.

``` bash
# Run pre-processing
python MLOD/scripts/preprocessing/gen_mini_batches.py

```
To configure the mini-batches, you can modify `mlod/configs/mb_preporcessing/rpn_class.config`.
You also need to select the *class* you want to train on. Inside the `gen_mini_batches.py` select what class to process.
By default it processes *Car* and *people* classes, where the flag `process_class` is set to True. The People class includes both
Pedestrian and Cyclists. You can also generate mini-batches for a single class such as Pedestrian only.

Note: This script does parallel processing with *x* number of processes for faster processing. This can also be disabled inside the script.

Once this script is done, you should now have the following folder inside `mlod`:


``` bash
# Mini-batches
cd mlod/data; ls
label_clusters mini_batches
```

### Training Configuration

There are sample configuration files for training inside `mlod/configs`. Rename the config file to your experiment name
and make sure the name matches the `checkpoint_name: 'mlod_fpn_car'` entry inside your config.

### Run Trainer

To start training, run the following:
``` bash
python mlod/experiments/models/run_training.py --pipeline_config=mlod/configs/mlod_fpn_car.config
```

(Optional) You can specify the gpu device you want to run it on by adding `--device='0'` to the command above.

### Run Evaluator

To start evaluation, run the following:
``` bash
python mlod/experiments/models/run_evaluation.py --pipeline_config=mlod/configs/mlod_fpn_car.config
```

The evaluator has two main modes, you can either evaluate a single checkpoint, a list of indices of checkpoints, or repeatedly.
By default, the evaluator is design to be ran in parallel with the trainer to repeatedly evaluate checkpoints. This can be configured
inside the same config file (look for `eval_config` entry).

Note: In addition to evaluating the loss, calclulating accuracies etc, the evaluator also runs the `kitti native evaluation code`.
This straightaway starts converting the predictions to Kitti format and calculates the `AP` for every checkpoint and saves the results inside `scripts/offline_eval/results/mlod_fpn_car_results_0.1.txt` where `0.1` is the score threshold.


### Run Inference

To run inference on the `test` split, use the following script:
``` bash
python mlod/experiments/models/run_testing.py --checkpoint_name='mlod_fpn_car' --data_split='test' --ckpt_indices=0
```
Here you can also use `val` split. The `ckpt_indices` here indicates the index of the checkpoint in the list. So say you set the
`checkpoint_interval` inside your config to `1000`. That means to evaluate say checkpoint `116000`, the index is `116`.
You can also just set this to `-1` where it evaluates the latest checkpoint.

### Visualization
All the results should go to `mlod/data/outputs`.

``` bash
cd mlod/data/outputs/mlod_fpn_car
```

Here you should see `proposals_and_scores` and `final_predictions_and_scores` results. To visualize these results, you can run
`demos/show_predictions_2d.py` and `demos/show_predictions_3d.py`. These scripts need to be configured to be ran on your experiments,
see the `demos/README.md`.

