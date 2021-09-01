from glob import glob
import os
import sys

import cv2
import numpy as np
from pycocotools.coco import COCO
from shapely.geometry import Polygon

sys.path.append(os.path.abspath("./"))
from util import background_color

base_dir = "/media/Synology3/Robert/splineDist/data/arena_pit"
image_dir = f"{base_dir}/images"
mask_dir = f"{base_dir}/masks"
coco_data = COCO(
    "/media/Synology3/Robert/splineDist/data/arena_pit/archive/arena_pits-17.json"
)
SCALING_FACTOR = 0.5
bbox_half_length = 80
for img_id in coco_data.imgs:
    filename_noext = os.path.splitext(coco_data.imgs[img_id]["file_name"])[0]
    filename_as_tif = filename_noext + ".tif"
    mask = cv2.cvtColor(
        cv2.imread(os.path.join(mask_dir, os.path.basename(filename_as_tif))),
        cv2.COLOR_BGR2GRAY,
    )
    img = cv2.imread(os.path.join(image_dir, os.path.basename(filename_as_tif)))
    print("opened image", img, mask)
    # all I want is to crop around each separate instance in the image.
    # an easier approach would be to just refer back to the original annotations.
    annotations = coco_data.getAnnIds(imgIds=[img_id])
    print("annotations:", annotations)
    for i, ann in enumerate(annotations):
        ann = coco_data.anns[ann]
        coords = [int(el * SCALING_FACTOR) for el in ann["segmentation"][0]]
        print(len(coords))
        coords = [
            tuple([coords[j * 2], coords[j * 2 + 1]])
            for j in range(int(len(coords) / 2))
        ]
        print("coords for polygon:", coords)
        print("original segmentation:", ann["segmentation"][0])
        polygon = Polygon(coords)
        centroid = polygon.centroid
        print("Polygon centroid:", polygon.centroid)
        print(filename_as_tif)
        print("image size:", img.shape)
        # cv2.drawMarker(img, (int(centroid.y), int(centroid.x)), (255, 0, 0))
        lbounds = [
            max(int(centroid.y) - bbox_half_length, 0),
            max(int(centroid.x) - bbox_half_length, 0),
        ]
        img_crop = img[
            lbounds[0] : int(centroid.y) + bbox_half_length,
            lbounds[1] : int(centroid.x) + bbox_half_length,
        ]
        mask_crop = mask[
            lbounds[0] : int(centroid.y) + bbox_half_length,
            lbounds[1] : int(centroid.x) + bbox_half_length,
        ]
        # cv2.imshow('debug', img_crop)
        # cv2.imshow('mask_debug', mask_crop)
        # cv2.waitKey(0)
        print("img crop shape:", img_crop.shape)
        print(
            "lower bounds:",
            int(centroid.y) - bbox_half_length,
            int(centroid.x) - bbox_half_length,
        )
        if np.any(np.array(img_crop.shape[:2]) < 2 * bbox_half_length):
            bgc = background_color(img_crop)
            print("Found undercrop")
            print("background color:", bgc)
            new_img = np.full(
                (2 * bbox_half_length, 2 * bbox_half_length, 3),
                fill_value=bgc,
                dtype=np.uint8,
            )
            new_img[
                2 * bbox_half_length - img_crop.shape[0] :,
                2 * bbox_half_length - img_crop.shape[1] :,
            ] = img_crop
            new_mask = np.zeros(
                (2 * bbox_half_length, 2 * bbox_half_length), dtype=np.uint8
            )
            new_mask[
                2 * bbox_half_length - img_crop.shape[0] :,
                2 * bbox_half_length - img_crop.shape[1] :,
            ] = mask_crop
            img_crop = new_img
            mask_crop = new_mask
            # cv2.imshow('debug', img_crop)
            # cv2.imshow('debug2', mask_crop)
            # cv2.waitKey(0)
        cv2.imwrite(
            os.path.join(base_dir, "patches", "images", f"{filename_noext}_{i}.tif"),
            img_crop,
        )
        cv2.imwrite(
            os.path.join(base_dir, "patches", "masks", f"{filename_noext}_{i}.tif"),
            mask_crop,
        )
        crop_2_ymin = np.random.randint(img.shape[0] - 2*bbox_half_length)
        crop_2_xmin = np.random.randint(img.shape[1] - 2*bbox_half_length)
        img_crop = img[
            crop_2_ymin:crop_2_ymin + 2 * bbox_half_length,
            crop_2_xmin:crop_2_xmin + 2 * bbox_half_length
        ]
        mask_crop = mask[
            crop_2_ymin:crop_2_ymin + 2*bbox_half_length,
            crop_2_xmin:crop_2_xmin + 2*bbox_half_length
        ]
        cv2.imwrite(
            os.path.join(base_dir, "patches", "images", f"{filename_noext}_rand_{i}.tif"),
            img_crop,
        )
        cv2.imwrite(
            os.path.join(base_dir, "patches", "masks", f"{filename_noext}_rand_{i}.tif"),
            mask_crop,
        )
