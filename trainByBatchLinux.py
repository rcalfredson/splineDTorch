import argparse
from net_retrain_manager import NetRetrainManager
import os
import subprocess
import shlex

# -d egg-fullsize -n FCRN_A -m P:\Robert\objects_counting_dmap\model_backup\egg_FCRN_A_expanded_dataset_v2.pth -lr 0.0045 -e 1 -hf 0.5 -vf 0.5 --plot --batch_size 4
# 2021-03-05
# python trainByBatchLinux.py 10 "--config configs/unet_backbone_rand_zoom.json --plot --val_interval 4 egg"


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


opts = options()
if opts.existing_nets:
    net_retrainer = NetRetrainManager(opts)
    while net_retrainer.nets_remaining_to_retrain():
        net_to_retrain = net_retrainer.get_random_net_to_retrain()
        run_one_training(net_to_retrain)
    exit()
for n in range(opts.nRepeats):
    subprocess.call(
        "python train.py --export_at_end %s" % opts.trainParams,
        shell=True,
        preexec_fn=os.setsid,
    )
