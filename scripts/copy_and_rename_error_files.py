from glob import glob
import os
from pathlib import Path
import shutil

sub_dirs = [
    f.path
    for f in os.scandir(
        r"P:\Robert\splineDist\data\egg\archive\to_compare_with_torch_nets"
    )
    if f.is_dir()
]
parent_dir = Path(sub_dirs[0]).parent
sub_dirs.remove(os.path.join(parent_dir, "temp"))
print(len(sub_dirs))
for d in sub_dirs:
    error_files = glob(os.path.join(d, "*_errors.json"))
    # print('files:', error_files)
    for f in error_files:
        print("parent dir:", parent_dir)
        dest = os.path.join(
            parent_dir,
            "temp",
            f"{os.path.basename(f.split('_errors')[0])}.pth_errors.json",
        )
        print("destination:", dest)
        # input()
        shutil.copy(f, dest)
