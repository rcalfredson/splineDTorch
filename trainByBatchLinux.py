import argparse
import os
import subprocess
import shlex

# -d egg-fullsize -n FCRN_A -m P:\Robert\objects_counting_dmap\model_backup\egg_FCRN_A_expanded_dataset_v2.pth -lr 0.0045 -e 1 -hf 0.5 -vf 0.5 --plot --batch_size 4
# 2021-03-05
# python trainByBatchLinux.py 5 '-d egg-unshuffled -n FCRN_A -lr 0.004 -e 3000 -hf 0.5 -vf 0.5 -rot --plot --batch_size 4 --config /media/Synology3/Robert/objects_counting_dmap/configs/shuffle_data_at_start_2021-02-23.json'

def options():
  """Parse options for the batch-mode FCRN-A training script."""
  p = argparse.ArgumentParser(description='Run multiple FCRN-A trainings' +\
    'in batch mode (i.e., in serial)')
  p.add_argument('nRepeats', help='number of times to repeat the training',
    type=int)
  p.add_argument('trainParams', help='options to pass to the training script' +\
    ' (note: enclose them in quotations)')
  return p.parse_args()

opts = options()
for n in range(opts.nRepeats):
  subprocess.call('python train.py --export_at_end %s'%opts.trainParams,
    shell=True,
    preexec_fn=os.setsid)
