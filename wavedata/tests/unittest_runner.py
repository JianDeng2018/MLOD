#!/usr/bin/env python3.5
import os
import re
import shutil
import subprocess
import sys


# SETTINGS
keep_unittest_logs = False
unittests_bin_dir = "tests"
unittests_log_dir = "unittests_log"
unittests_file_pattern = "test_[a-zA-Z0-9_]*.py"


class TC:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def print_stdout(file_path):
    with open(file_path, 'r') as f:
        print("{}\n{}\n{}".format("-" * 79, f.read(), "-" * 79))


def get_files(path, pattern):
    file_list = []

    for root, directory, files in os.walk(path):
        for f in files:
            if re.match(pattern, f):
                file_list.append(os.path.join(root, f))

    return file_list

if __name__ == "__main__":
    orig_cwd = os.getcwd()

    # make log dir if not already exist
    if not os.path.exists(unittests_log_dir):
        os.mkdir(unittests_log_dir)

    # execute all unittests
    error = False
    for unittest in get_files(unittests_bin_dir, unittests_file_pattern):
        print("UNITTEST [{}] ".format(unittest))
        sys.path.append(os.path.abspath(os.path.dirname(unittest)))

        unittest_output_fp = os.path.join(
            unittests_log_dir, os.path.basename(unittest) + ".log")

        # execute unittest
        try:
            with open(unittest_output_fp, 'w') as f:
                subprocess.check_call([
                    "coverage", "run", "--source=tools",
                    "-p", "./{}".format(unittest)],
                    stdout=f, stderr=f)

            print("{}PASSED!{}".format(TC.OKGREEN, TC.ENDC))
        except:
            print("{}FAILED!{}".format(TC.FAIL, TC.ENDC))
            print_stdout(unittest_output_fp)
            error = True
    # keep unittest stdout dir?
    if not keep_unittest_logs:
        shutil.rmtree(unittests_log_dir)

    # combine coverage data
    subprocess.call(["coverage", "combine"])
    subprocess.call(["coverage", "report"])
    subprocess.call(["coverage", "html"])
    os.chdir(orig_cwd)
    if error is True:
        sys.exit(-1)
    else:
        sys.exit(0)
