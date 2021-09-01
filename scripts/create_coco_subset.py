from pycocotools.coco import COCO
from glob import glob
import json
import os

coco_path = "/media/Synology3/Robert/splineDist/data/arena_pit/archive/arena_pits-17.json"
coco_data = COCO(
    coco_path
)
with open(coco_path) as f:
    coco_json = json.load(f)
imgs = [
    os.path.basename(fn)
    for fn in glob(
        "/media/Synology3/Robert/splineDist/data/arena_pit/patches/heldout/images/*tif"
    )
]
print('Num images before pruning:', len(coco_json['images']))
for img_id in coco_data.imgs:
    file_name = os.path.splitext(coco_data.imgs[img_id]["file_name"])[0] + ".tif"
    if file_name not in imgs:
        print('file name', file_name,'not in list:')
        print(imgs)
        coco_json['images'].remove(coco_data.imgs[img_id])
print("Num images after pruning:", len(coco_json['images']))
