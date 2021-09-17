import argparse
from net_retrain_manager import NetRetrainManager
import os
import subprocess
import shlex
import sys

# python trainByBatchLinux.py --n_repeats 10 "--config configs/unet_backbone_rand_zoom.json --plot --val_interval 4 egg" > /dev/null 2>&1 &


def options():
    """Parse options for the batch-mode FCRN-A training script."""
    p = argparse.ArgumentParser(
        description="Run multiple FCRN-A trainings" + "in batch mode (i.e., in serial)"
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--n_repeats",
        help="number of times to repeat the training."
        " If using a --train_param_list, then the experiment on"
        " each line will be run n_repeats number of times.",
        type=int,
    )
    group.add_argument(
        "--existing_nets", help="Path to folder containing existing nets to retrain."
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--train_params",
        help="options to pass to the training script"
        + " (note: enclose them in quotations)",
    )
    group.add_argument(
        "--train_params_list",
        help="path to text file containing newline-separated list of experiments to run,"
        " i.e., each line of the file contains the arguments to pass to the training script.",
    )
    return p.parse_args()


def run_one_training(train_params, existing_model=None):
    subprocess.call(
        "python train.py --export_at_end %s%s"
        % (
            train_params,
            ""
            if existing_model is None
            else f' -m "{os.path.join(opts.existing_nets, existing_model)}"',
        ),
        shell=True,
        preexec_fn=os.setsid,
    )


opts = options()
if opts.existing_nets:
    if opts.train_params_list is not None:
        sys.exit(
            "Can only use one set of training options if retraining a group"
            " of existing nets."
        )
    net_retrainer = NetRetrainManager(opts.existing_nets)
    while net_retrainer.nets_remaining_to_retrain():
        net_to_retrain = net_retrainer.get_random_net_to_retrain()
        run_one_training(opts.train_params, net_to_retrain)
    exit()

if opts.train_params:
    for _ in range(opts.n_repeats):
        run_one_training(opts.train_params)
elif opts.train_params_list:
    with open(opts.train_params_list, "r") as f:
        train_params = f.read().splitlines()
    for experiment_config in train_params:
        for _ in range(opts.n_repeats):
            run_one_training(experiment_config)
