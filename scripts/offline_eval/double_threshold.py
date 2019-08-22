import os
import sys
import numpy as np
from mlod.core import box_3d_projector
from wavedata.tools.core import calib_utils
from PIL import Image
import mlod
from multiprocessing import Process
import subprocess

def get_rgb_image_path(sample_name):
    rgb_image_dir = '/home/j7deng/Kitti/object/training/image_2'
    return rgb_image_dir + '/' + sample_name + '.png'

def run_kitti_native_script(checkpoint_name, score_threshold, global_step, sub_score_threshold):
    """Runs the kitti native code script."""

    eval_script_dir = mlod.root_dir() + '/data/outputs/' + \
        checkpoint_name + '/predictions'
    make_script = eval_script_dir + \
        '/kitti_native_eval/run_eval.sh'
    script_folder = eval_script_dir + \
        '/kitti_native_eval/'
    results_dir = mlod.top_dir() + '/scripts/offline_eval/results/'

    # Round this because protobuf encodes default values as full decimal
    score_threshold = round(score_threshold, 3)
    print('evaluating....')

    subprocess.call([make_script, script_folder,
                     str(score_threshold),
                     str(global_step),
                     str(checkpoint_name),
                     str(results_dir),
                     sub_score_threshold])

def save_predictions_in_kitti_format(sample_names,
                                     checkpoint_name,
                                     final_predictions_root_dir,
                                     branch_predictions_root_dirs,
                                     kitti_predictions_3d_dir,
                                     score_threshold,
                                     sub_score_threshold,
                                     global_step):
    """ Converts a set of network predictions into text files required for
    KITTI evaluation.
    """

    # Round this because protobuf encodes default values as full decimal
    score_threshold = round(score_threshold, 3)

    # Get available prediction folders
    predictions_root_dir = mlod.root_dir() + '/data/outputs/' + \
        checkpoint_name + '/predictions'

    final_predictions_dir = final_predictions_root_dir + \
        '/' + str(global_step)
    branch_prediction_dir_list = [branch_dir + '/' + str(global_step) for branch_dir in branch_predictions_root_dirs]

    if not os.path.exists(kitti_predictions_3d_dir):
        os.makedirs(kitti_predictions_3d_dir)

    # Do conversion
    num_samples = len(sample_names)
    num_valid_samples = 0

    print('\nGlobal step:', global_step)
    print('Converting detections from:', final_predictions_dir)

    print('3D Detections being saved to:', kitti_predictions_3d_dir)

    for sample_idx,sample_name in enumerate(sample_names):

        # Print progress
        sys.stdout.write('\rConverting {} / {}'.format(
            sample_idx + 1, num_samples))
        sys.stdout.flush()

        prediction_file = sample_name + '.txt'

        kitti_predictions_3d_file_path = kitti_predictions_3d_dir + \
            '/' + prediction_file

        predictions_file_path = final_predictions_dir + \
            '/' + prediction_file

        # If no predictions, skip to next file
        if not os.path.exists(predictions_file_path):
            np.savetxt(kitti_predictions_3d_file_path, [])
            continue

        all_predictions = np.loadtxt(predictions_file_path)

        sub_predictions_list = []
        for branch_prediction_dir in branch_prediction_dir_list:
            sub_predictions_file_path = branch_prediction_dir + \
                '/' + prediction_file
            sub_predictions = np.loadtxt(sub_predictions_file_path)
            sub_predictions_list.append(sub_predictions)

        # # Swap l, w for predictions where w > l
        # swapped_indices = all_predictions[:, 4] > all_predictions[:, 3]
        # fixed_predictions = np.copy(all_predictions)
        # fixed_predictions[swapped_indices, 3] = all_predictions[
        #     swapped_indices, 4]
        # fixed_predictions[swapped_indices, 4] = all_predictions[
        #     swapped_indices, 3]

        score_filter = np.array(all_predictions[:, 7] >= score_threshold )
        #print('main',score_filter)
        score_filter1 = np.array(sub_predictions_list[0][:,7] >= sub_score_threshold[0])
        #print('br0',score_filter1)
        score_filter2 = np.array(sub_predictions_list[1][:,7] >= sub_score_threshold[1])
        #print('br1',score_filter2)

        all_predictions = all_predictions[score_filter & score_filter1 & score_filter2]

        # If no predictions, skip to next file
        if len(all_predictions) == 0:
            np.savetxt(kitti_predictions_3d_file_path, [])
            continue

        # Project to image space
        sample_name = prediction_file.split('.')[0]
        img_idx = int(sample_name)

        # Load image for truncation
        image = Image.open(get_rgb_image_path(sample_name))

        calib_dir = '/home/j7deng/Kitti/object/training/calib'
        stereo_calib_p2 = calib_utils.read_calibration(calib_dir,
                                                       img_idx).p2

        boxes = []
        image_filter = []
        for i in range(len(all_predictions)):
            box_3d = all_predictions[i, 0:7]
            img_box = box_3d_projector.project_to_image_space(
                box_3d, stereo_calib_p2,
                truncate=True, image_size=image.size)

            # Skip invalid boxes (outside image space)
            if img_box is None:
                image_filter.append(False)
                continue

            image_filter.append(True)
            boxes.append(img_box)

        boxes = np.asarray(boxes)
        all_predictions = all_predictions[image_filter]

        # If no predictions, skip to next file
        if len(boxes) == 0:
            np.savetxt(kitti_predictions_3d_file_path, [])
            continue

        num_valid_samples += 1

        # To keep each value in its appropriate position, an array of zeros
        # (N, 16) is allocated but only values [4:16] are used
        kitti_predictions = np.zeros([len(boxes), 16])

        # Get object types
        all_pred_classes = all_predictions[:, 8].astype(np.int32)
        dataset_classes = ['Pedestrian', 'Cyclist']
        obj_types = [dataset_classes[class_idx]
                     for class_idx in all_pred_classes]

        # Truncation and Occlusion are always empty (see below)

        # Alpha (Not computed)
        kitti_predictions[:, 3] = -10 * np.ones((len(kitti_predictions)),
                                                dtype=np.int32)

        # 2D predictions
        kitti_predictions[:, 4:8] = boxes[:, 0:4]

        # 3D predictions
        # (l, w, h)
        kitti_predictions[:, 8] = all_predictions[:, 5]
        kitti_predictions[:, 9] = all_predictions[:, 4]
        kitti_predictions[:, 10] = all_predictions[:, 3]
        # (x, y, z)
        kitti_predictions[:, 11:14] = all_predictions[:, 0:3]
        # (ry, score)
        kitti_predictions[:, 14:16] = all_predictions[:, 6:8]

        # Round detections to 3 decimal places
        kitti_predictions = np.round(kitti_predictions, 3)

        # Empty Truncation, Occlusion
        kitti_empty_1 = -1 * np.ones((len(kitti_predictions), 2),
                                     dtype=np.int32)

        # Stack 3D predictions text
        kitti_text_3d = np.column_stack([obj_types,
                                         kitti_empty_1,
                                         kitti_predictions[:, 3:16]])

        # Save to text files
        np.savetxt(kitti_predictions_3d_file_path, kitti_text_3d,
                   newline='\r\n', fmt='%s')

    print('\nNum valid:', num_valid_samples)
    print('Num samples:', num_samples)

def main():
    base_dir = '/media/j7deng/Data/track/my-mlod/mlod/data/outputs/mlod_fpn_people/predictions/kitti_native_eval/0.1/'
    checkpoint_name = 'mlod_fpn_people'
    final_predictions_root_dir = '/media/j7deng/Data/track/my-mlod/mlod/data/outputs/mlod_fpn_people/predictions/final_predictions_and_scores/val'
    branch_predictions_root_dirs = ['/media/j7deng/Data/track/my-mlod/mlod/data/outputs/mlod_fpn_people/predictions/final_predictions_and_scores_0/val',
                                    '/media/j7deng/Data/track/my-mlod/mlod/data/outputs/mlod_fpn_people/predictions/final_predictions_and_scores_1/val']
    sub_score_threshold = [0.3,0.1]
    score_threshold = 0.1
    global_step = 24000
    kitti_predictions_3d_dir = '/media/j7deng/Data/track/my-mlod/mlod/data/outputs/mlod_fpn_people/predictions/kitti_native_eval/0.1/01_01/' + str(global_step) + '/data'
    sub_score_threshold_str = '01_01'
    # read sample names
    set_file = '/home/j7deng/Kitti/object/val.txt'
    with open(set_file, 'r') as f:
        sample_names = f.read().splitlines()
    
    save_predictions_in_kitti_format(sample_names,
                                     checkpoint_name,
                                     final_predictions_root_dir,
                                     branch_predictions_root_dirs,
                                     kitti_predictions_3d_dir,
                                     score_threshold,
                                     sub_score_threshold,
                                     global_step)

    run_kitti_native_script(checkpoint_name, score_threshold, global_step,sub_score_threshold_str)

if __name__=='__main__':
    main()
