from glob import glob
import json
import os
import re

name_maps = glob(
    "/media/Synology3/Robert/splineDTorch/saved_models/"
    "egg/unet_expanded_data_1_1600epch/*metadata*.json"
)
map_data = {}

print("name maps:", name_maps)
for map_filename in name_maps:
    print('map filename:', map_filename)
    with open(map_filename) as f:
        net_data = json.load(f)
    # map_data[os.path.basename(map_filename)]
    sub_start = "_meta"
    sub_end = "data"
    model_name = (
        re.sub(
            r"{}.*?{}".format(re.escape(sub_start), re.escape(sub_end)),
            "_400epochs",
            os.path.basename(map_filename),
        ).split(".json")[0]
        + ".pth"
    )
    print("is this correct model name?", model_name)
    print(
        os.path.exists(
            os.path.join(
                "/media/Synology3/Robert/splineDTorch/saved_models/egg/unet_expanded_data_1_1600epch/complete_nets",
                model_name,
            )
        )
    )
    map_data[model_name] = os.path.basename(net_data["starter_model"])
with open('/media/Synology3/Robert/splineDTorch/saved_models/egg/unet_expanded_data_1_1600epch/complete_nets/name_map.json', 'w') as f:
    json.dump(map_data, f, ensure_ascii=False, indent=4)