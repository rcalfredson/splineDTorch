from glob import glob
import os
import shutil

folders = "/media/Synology3/Rebecca/2021_7_8,/media/Synology3/Rebecca/2021_7_10_B,/media/Synology3/Rebecca/2021_7_13,/media/Synology3/Rebecca/2021_7_14,/media/Synology3/Rebecca/2021_7_15,/media/Synology3/Rebecca/2021_7_16,/media/Synology3/Rebecca/2021_7_17,/media/Synology3/Rebecca/2021_7_18,/media/Synology3/Rebecca/2021_7_24,/media/Synology3/Rebecca/2021_7_25,/media/Synology3/Rebecca/2021_7_26,/media/Synology3/Rebecca/2021_7_27,/media/Synology3/Rebecca/2021_7_28,/media/Synology3/Rebecca/2021_7_29,/media/Synology3/Rebecca/2021_7_30,/media/Synology3/Rebecca/2021_8_1,/media/Synology3/Rebecca/2021_8_2,/media/Synology3/Rebecca/2021_8_3,/media/Synology3/Rebecca/2021_8_4,/media/Synology3/Rebecca/2021_8_8,/media/Synology3/Rebecca/2021_8_9,/media/Synology3/Rebecca/2021_8_13,/media/Synology3/Rebecca/2021_8_16,/media/Synology3/Rebecca/2021_8_17,/media/Synology3/Rebecca/2021_8_19,/media/Synology3/Rebecca/2021_5_15,/media/Synology3/Rebecca/2021_5_16,/media/Synology3/Rebecca/2021_5_17,/media/Synology3/Rebecca/2021_5_18,/media/Synology3/Rebecca/2021_5_19,/media/Synology3/Rebecca/2021_5_20,/media/Synology3/Rebecca/2021_5_21,/media/Synology3/Rebecca/2021_5_22,/media/Synology3/Rebecca/2021_5_23,/media/Synology3/Rebecca/2021_5_24".split(
    ","
)

destination = '/home/robert/coco-annotator/datasets/arena-wells-eval'

print('folders:', folders)
for folder in folders:
    imgs = glob(os.path.join(folder, '*.JPG'))
    for img in imgs:
        path_parts = img.split(os.sep)
        new_filename = f"{path_parts[-2]}_{path_parts[-1]}"
        shutil.copy(img, os.path.join(destination,
            new_filename))
