import argparse
from glob import glob
import json
import os

p = argparse.ArgumentParser(
    description="collect the names of images in a given folder, then search for and"
    " delete images (and accompanying annotations) in a given COCO file whose"
    " filenames partially match those in the folder."
)

p.add_argument(
    "image_dir",
    help="folder containing images which should be removed from the COCO file.",
)

p.add_argument("coco_file", help="COCO file to modify")

p.add_argument(
    "--dest",
    "-d",
    help="path at which to save the modified COCO file. By default, will create an"
    ' adjacent copy with the suffix "trimmed".',
    required=False,
)

opts = p.parse_args()
extensions = ("png", "jpg", "tif")
images = []
for ext in extensions:
    images += [
        os.path.splitext(os.path.basename(filename))[0]
        for filename in glob(f"{opts.image_dir}/*.{ext}")
    ]
print(images)
with open(opts.coco_file) as f:
    coco_file = json.load(f)

# print(coco_file)
for img in images:
    print(f"Checking {img}")
    match = next(
        (item for item in coco_file["images"] if img in item["file_name"]), None
    )
    print(f"Found match? {match}")
    if match is not None:
        coco_file['images'].remove(match)
    
    img_id = match['id']
    annot_match = 0
    while annot_match is not None:
        annot_match = next(
            ()
        )
