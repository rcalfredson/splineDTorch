import argparse
from net_retrain_manager import NetRetrainManager
import os
import subprocess
import shlex

# python trainByBatchLinux.py --n_repeats 10 "--config configs/unet_backbone_rand_zoom.json --plot --val_interval 4 egg --coco_file_path /media/Synology3/Robert/amazon_annots/coco/consolidated_all.json" > /dev/null 2>&1 &


def options():
    """Parse options for the batch-mode FCRN-A training script."""
    p = argparse.ArgumentParser(
        description="Run multiple FCRN-A trainings" + "in batch mode (i.e., in serial)"
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--n_repeats", help="number of times to repeat the training", type=int
    )
    group.add_argument(
        "--existing_nets", help="Path to folder containing existing nets to retrain."
    )
    p.add_argument(
        "trainParams",
        help="options to pass to the training script"
        + " (note: enclose them in quotations)",
    )
    return p.parse_args()


def run_one_training(existing_model=None):
    subprocess.call(
        "python train.py --export_at_end %s%s"
        % (
            opts.trainParams,
            ""
            if existing_model is None
            else f' -m "{os.path.join(opts.existing_nets, existing_model)}"',
        ),
        shell=True,
        preexec_fn=os.setsid,
    )


opts = options()
if opts.existing_nets:
    net_retrainer = NetRetrainManager(opts.existing_nets)
    while net_retrainer.nets_remaining_to_retrain():
        net_to_retrain = net_retrainer.get_random_net_to_retrain()
        run_one_training(net_to_retrain)
    exit()
for n in range(opts.n_repeats):
    run_one_training()
