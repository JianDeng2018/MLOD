from mlod.builders import config_builder_util
from mlod.builders.dataset_builder import DatasetBuilder


def set_up_video_dataset(dataset_config,
                         dataset_dir,
                         data_split,
                         data_split_dir):

    # Overwrite fields
    dataset_config.name = 'kitti_video'
    dataset_config.dataset_dir = dataset_dir
    dataset_config.data_split = data_split
    dataset_config.data_split_dir = data_split_dir
    dataset_config.has_labels = False

    # Overwrite repeated fields
    dataset_config = config_builder_util.proto_to_obj(dataset_config)
    dataset_config.aug_list = []

    dataset = DatasetBuilder.build_kitti_dataset(dataset_config,
                                                 use_defaults=False)

    return dataset
