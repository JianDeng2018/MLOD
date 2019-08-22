import os


def main():
    """Merges predictions for multiple outputs into a 'merged' folder
    """

    merged_folder = 'merged'
    if os.path.exists(merged_folder):
        raise ValueError('Merged folder exists already. '
                         'Delete or rename it first')

    os.makedirs(merged_folder)

    kitti_pred_folders = [
        # Cars
        'path_to_car_detections/data',

        # People
        'path_to_people_detections/data',
    ]

    pred_folder = kitti_pred_folders[0]

    file_names = sorted(os.listdir(pred_folder))

    # Create new text file for each file
    for file_name in file_names:

        # Create merged text file
        merged_text_path = merged_folder + '/' + file_name
        with open(merged_text_path, 'a') as merged_file:

            for pred_folder in kitti_pred_folders:
                file_path = pred_folder + '/' + file_name

                if os.path.exists(file_path):
                    # Append to new text file if it exists
                    with open(file_path) as pred_file:
                        contents = pred_file.read()
                        if contents:
                            merged_file.write(contents)

                else:
                    raise ValueError('Missing file {} in {}'.format(
                        file_name, pred_folder))


if __name__ == '__main__':
    main()
