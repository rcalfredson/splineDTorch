import argparse
from glob import glob
import json
import os
from statistics import ttest_ind, ttest_rel

import numpy as np

parser = argparse.ArgumentParser(
    description="Run t-test comparison of results from" " two SplineDist learning experiments"
)
parser.add_argument("dir1", help="Folder containing first set of error results")
parser.add_argument("dir2", help="Folder containing second set of error results")
parser.add_argument(
    "--paired_name_map",
    help="JSON file mapping names of original nets to their paired counterparts as"
    " part of a dependent t-test for paired samples. The keys of the dictionary"
    " should correspond to nets from dir1, and the values to nets from dir2.",
)
opts = parser.parse_args()

error_categories = (
    "mean_abs_error",
    "mean_rel_error",
    "mean_rel_error_0to10",
    "mean_rel_error_11to40",
    "mean_rel_error_41plus",
    "max_abs_error",
)
errors = [{k: [] for k in error_categories} for _ in range(2)]
outliers = [[] for _ in range(2)]


if opts.paired_name_map:
    with open(opts.paired_name_map) as f:
        paired_name_map = json.load(f)
    pairing_order = []


def process_single_error(i, error_filename, outliers_for_dir):
    if outliers_for_dir and os.path.basename(error_filename) in outliers_for_dir:
        return
    with open(error_filename, "r") as f:
        single_net_errors = json.load(f)
        for err_line in single_net_errors:
            if err_line == "errors_by_img":
                continue
            errors[i][err_line].append(float(single_net_errors[err_line]))


def folders_in(path_to_parent):
    for fname in os.listdir(path_to_parent):
        if os.path.isdir(os.path.join(path_to_parent, fname)):
            yield os.path.join(path_to_parent, fname)

def parse_error_for_dir(dir_name, index):
    error_files = glob(os.path.join(dir_name, "*_errors.json"))
    print('error files:', error_files)
    for sub_folder in folders_in(dir_name):
        error_files += glob(os.path.join(sub_folder, "*_errors.json"))
    outlier_filename = os.path.join(dir_name, "outliers.txt")
    if os.path.isfile(outlier_filename):
        with open(outlier_filename, "r") as f:
            outliers_for_dir = ["%s.txt" % line for line in f.read().splitlines()]
    else:
        outliers_for_dir = None
    if not opts.paired_name_map or opts.paired_name_map and index == 0:
        for error_file in error_files:
            print('processing a single error')
            process_single_error(index, error_file, outliers_for_dir)
            if opts.paired_name_map:
                pairing_order.append(os.path.basename(error_file).split("_errors")[0])
    elif opts.paired_name_map and index > 0:
        for net_name in pairing_order:
            error_file = "%s_errors.json" % (paired_name_map[net_name])
            process_single_error(
                index, os.path.join(dir_name, error_file), outliers_for_dir
            )

def parse_errors():
    for i, dir_name in enumerate((opts.dir1, opts.dir2)):
        print('dir name:', dir_name)
        parse_error_for_dir(dir_name, i)


parse_errors()


for err_cat in error_categories:
    print("Error category:", err_cat)
    if opts.paired_name_map:
        fn_to_call = ttest_rel
    else:
        fn_to_call = ttest_ind
    res = fn_to_call(
        np.asarray(errors[0][err_cat]), np.asarray(errors[1][err_cat]), msg=True
    )
    print()