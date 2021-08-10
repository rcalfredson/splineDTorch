import argparse
import platform
import subprocess

if platform.node() == "Yang-Lab-Dell2":
    condaEnv = "detectronEnv"
    driveLetter = "R"
else:
    condaEnv = "pytorchEnv"
    driveLetter = "P"


def options():
    """Parse options for the batch-mode FCRN-A training script."""
    p = argparse.ArgumentParser(
        description="Run multiple FCRN-A trainings" + "in batch mode (i.e., in serial)"
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--n_repeats",
        help="Number of times to repeat the training.",
        type=int,
    )
    group.add_argument(
        "--existing_nets", help="Path to folder containing existing nets to retrain."
    )
    p.add_argument(
        "trainParams",
        help="Options to pass to the training script"
        + " (note: enclose them in quotations)",
    )
    return p.parse_args()


opts = options()


def run_one_training():
    command_being_called = (
        'start /w %%windir%%\\System32\\cmd.exe "/K" C:\\Users\\Tracking\\anaconda3\\Scripts\\activate.bat C:\\Users\\Tracking\\anaconda3 ^& conda activate %s ^& %s: ^& cd Robert\\objects_counting_dmap ^& cd ^& python train.py --export_at_end %s ^& exit'
        % (condaEnv, driveLetter, opts.trainParams)
    )

    subprocess.call(
        command_being_called,
        shell=True,
    )


if opts.existing_nets:
    net_retrainer = NetRetrainManager(opts)
    while net_retrainer.nets_remaining_to_retrain():
        net_to_retrain = net_retrainer.get_random_net_to_retrain()
        run_one_training(net_to_retrain)
    exit()
for _ in range(opts.n_repeats):
    run_one_training()
