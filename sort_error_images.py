from glob import glob
import os
from pathlib import Path
import shutil

exp_47_imgs = glob(
    "/media/Synology3/Robert/splineDTorch/saved_models/egg/unet_expanded_data_1/complete_nets/error_examples/*.png"
)
exp_49_imgs = glob(
    "/media/Synology3/Robert/splineDTorch/saved_models/egg/unet_blur_disabled/complete_nets/error_examples/*.png"
)

print(exp_47_imgs)
print(exp_49_imgs)
print(os.sep)
# first step: eliminate any image names that aren't common to both lists.
by_img_name = {}
for k, l in (("47", exp_47_imgs), ("49", exp_49_imgs)):
    by_img_name[k] = {}
    for img in l:
        img_name = "_".join(
            img.split(os.sep)[-1].split("_splinedist")[0].split("_")[1:]
        )
        by_img_name[k][img_name] = img
# go through one of the dictionaries, and if the key is not in the other,
# then delete the key.
print(by_img_name)
print(f"Number keys at start: {len(by_img_name['47'].keys())}")
print(len(by_img_name["49"].keys()))
# for img_name in list(by_img_name["47"]):
# if img_name not in by_img_name["49"].keys():
# del by_img_name["47"][img_name]
# for img_name in list(by_img_name["49"]):
# if img_name not in by_img_name["47"].keys():
# del by_img_name["49"][img_name]
print(
    "and after pruning:", len(by_img_name["47"].keys()), len(by_img_name["47"].keys())
)

for img_name in list(by_img_name["47"]):
    if (
        img_name in by_img_name["49"].keys()
        and by_img_name["47"][img_name].split(os.sep)[-1].split("_")[0]
        == by_img_name["49"][img_name].split(os.sep)[-1].split("_")[0]
    ):
        del by_img_name["47"][img_name]
        del by_img_name['49'][img_name]
for img_name in list(by_img_name["49"]):
    if (
        img_name in by_img_name["47"].keys()
        and by_img_name["47"][img_name].split(os.sep)[-1].split("_")[0]
        == by_img_name["49"][img_name].split(os.sep)[-1].split("_")[0]
    ):
        print('deleting in the second block')
        del by_img_name['49'][img_name]
        del by_img_name['47'][img_name]
print(
    "Number keys at end:", len(by_img_name["47"].keys()), len(by_img_name["49"].keys())
)
for exp_number in ('47', '49'):
    for k in by_img_name[exp_number]:
        shutil.copy(by_img_name[exp_number][k], os.path.join(Path(by_img_name[exp_number][k]).parent, 
            'for_compare', os.path.basename(by_img_name[exp_number][k])))
    
# print("For exp 47:", by_img_name["47"])
# print()
# print("For exp 49:", by_img_name["49"])
