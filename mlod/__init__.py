import os


def root_dir():
    return os.path.dirname(os.path.realpath(__file__))


def top_dir():
    mlod_root_dir = root_dir()
    return os.path.split(mlod_root_dir)[0]
