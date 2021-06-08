import argparse
from glob import glob
import os
from pathlib import Path
import platform
import shutil
import subprocess

parser = argparse.ArgumentParser(
    description="Wrapper script for" " running the eval script for multiple experiments"
)
parser.add_argument(
    "eval_params",
    help="Options to pass the eval script (" "note: enclose them in quotation marks)",
)

parser.add_argument('dataPath', help="Path to folder containing eval images")

parser.add_argument(
    "exp_list",
    help="Path to newline-separated file of"
    " experiment directories (can be relative to the current path). Note:"
    " nets to be measured are expected to be in a sub-folder named"
    " complete_nets.",
)
parser.add_argument(
    "result_dir",
    help="Name of sub-folder inside the"
    " experiment folder where the results will be copied at the end of"
    " running the evaluation script (folder will be created if necessary)",
)
parser.add_argument(
    "--config_list",
    help="Path to newline-separated file of config files (can be relative to"
    " the current path). Note: should be listed in corresponding order"
    " with the experiments. Default path is configs/defaults.json.",
)

opts = parser.parse_args()
with open(opts.exp_list, "r") as f:
    experiments = f.read().splitlines()
if opts.config_list:
    with open(opts.config_list) as f:
        configs = f.read().splitlines()

for i, exp in enumerate(experiments):
    # the script seems to lack the argument to the folder containing the eval
    # images.
    command = (
        f"python eval_via_pt_data.py \"{opts.dataPath}\" \"{os.path.join(exp, 'complete_nets', '*.pth')}\""
        + f" {opts.eval_params}"
    )
    command += f" --dest_dir {opts.result_dir}"
    if opts.config_list:
        command += f" --config {configs[i]}"
    print('Calling the eval script with this command:', command)
    subprocess.call(command, shell=True)
    results_source_dir = f"error_results_{platform.node()}"
    # results_files = os.listdir(results_source_dir)
    results_files = glob(os.path.join(results_source_dir, "*.json"))
    dest_dir = os.path.join(exp, opts.result_dir)
    Path(dest_dir).mkdir(exist_ok=True, parents=True)
    for file_name in results_files:
        shutil.move(file_name, dest_dir)

# temp command: python eval_batch.py "" experiments_to_measure.txt task1-uli-errors
# python scripts/eval_batch.py "" /media/Synology3/Robert/objects_counting_dmap/egg_source/heldout_uli_WT_1  experiments_to_measure.txt task1-uli-errors --config_list configs_for_experiments.txt
