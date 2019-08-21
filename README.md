# Aggregate View Object Detection
[![Build Status](https://travis-ci.com/wavelab/avod-dev.svg?token=EadsqWkUzKDHZjRZYta4&branch=master)][1]

[1]: https://travis-ci.com/wavelab/avod-dev
This repository contains the implementation of our Aggregate View Object Detection network for 3D object detection.

## Getting Started
Implemented and tested on Ubuntu 16.04 and Python 3.5.

1. Install [Wavedata](https://github.com/wavelab/wavedata) dependencies

2. Install [Tensorflow](https://www.tensorflow.org/)


3. Git requires explicit download of the submodule's content. You can use :
``` bash
git submodule update --init --recursive
```

If cloning for the first time could also use the `clone` command.

4. Add wavedata to PYTHONPATH

``` bash
# From avod/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/wavedata
```

Note: This command needs to run from every new terminal you start. If you wish
to avoid running this manually, you can add it to your ~/.bashrc file.

5. Install Avod
``` bash
# From avod/
python setup.py install
```

### Protobuf Compilation

Avod uses Protobufs to configure model and
training parameters. Before the framework can be used, the Protobuf libraries
must be compiled. This should be done by running the following script:


``` bash
# From top level avod folder
sh avod/protos/run_protoc.sh
```

Alternatively, you run `protoc` command directly:

``` bash
# From top level avod folder
protoc avod/protos/*.proto --python_out=.
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
cd avod
python scripts/preprocessing/gen_mini_batches.py

```
To configure the mini-batches, you can modify `avod/configs/mb_preporcessing/rpn_class.config`.
You also need to select the *class* you want to train on. Inside the `gen_mini_batches.py` select what class to process.
By default it processes *Car* and *people* classes, where the flag `process_class` is set to True. The People class includes both
Pedestrian and Cyclists. You can also generate mini-batches for a single class such as Pedestrian only.

Note: This script does parallel processing with *x* number of processes for faster processing. This can also be disabled inside the script.

Once this script is done, you should now have the following folder inside `avod`:


``` bash
# Mini-batches
cd avod/data; ls
label_clusters mini_batches
```

### Training Configuration

There are sample configuration files for training inside `avod/configs`. You can train on the vanilla config, or modify an existing configuration.
To train a new configuration, copy a vanilla config, say `avod_exp_example.config`, rename this file to your experiment name
and make sure the name matches the `checkpoint_name: 'avod_exp_example'` entry inside your config.

### Run Trainer

To start training, run the following:
``` bash
python avod/experiments/models/run_training.py --pipeline_config=avod/configs/avod_exp_example.config
```

(Optional) You can specify the gpu device you want to run it on by adding `--device='0'` to the command above.

### Run Evaluator

To start evaluation, run the following:
``` bash
python avod/experiments/models/run_evaluation.py --pipeline_config=avod/configs/avod_exp_example.config
```

The evaluator has two main modes, you can either evaluate a single checkpoint, a list of indices of checkpoints, or repeatedly.
By default, the evaluator is design to be ran in parallel with the trainer to repeatedly evaluate checkpoints. This can be configured
inside the same config file (look for `eval_config` entry).

Note: In addition to evaluating the loss, calclulating accuracies etc, the evaluator also runs the `kitti native evaluation code`.
This straightaway starts converting the predictions to Kitti format and calculates the `AP` for every checkpoint and saves the results inside `scripts/offline_eval/results/avod_exp_example_results_0.1.txt` where `0.1` is the score threshold.


### Run Inference

To run inference on the `test` split, use the following script:
``` bash
python avod/experiments/models/run_testing.py --checkpoint_name='avod_exp_example' --data_split='test' --ckpt_indices=0
```
Here you can also use `val` split. The `ckpt_indices` here indicates the index of the checkpoint in the list. So say you set the
`checkpoint_interval` inside your config to `1000`. That means to evaluate say checkpoint `116000`, the index is `116`.
You can also just set this to `-1` where it evaluates the latest checkpoint.

### Visualization
All the results should go to `avod/data/outputs`.

``` bash
cd avod/data/outputs/avod_exp_experiment
```

Here you should see `proposals_and_scores` and `final_predictions_and_scores` results. To visualize these results, you can run
`demos/show_predictions_2d.py` and `demos/show_predictions_3d.py`. These scripts need to be configured to be ran on your experiments,
see the `demos/README.md`.




## LICENSE

Copyright (c) <2017> <Wavelab>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
