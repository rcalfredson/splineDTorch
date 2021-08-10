# Run the evaluation script without requiring instance segmentation for the
# input data

import argparse
import copy
from csbdeep.utils import normalize
import cv2
from glob import glob
import json
import matplotlib

import math

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
import random
from shapely.geometry import Polygon
from splinedist.config import Config
from splinedist.models.model2d import SplineDist2D
from splinedist.plot.plot import _draw_polygons
from splinedist.utils import data_dir
import spline_generator as sg
import sys

sys.path.append(os.path.abspath("../counting-3"))
import timeit
import torch
from util import COL_W, COL_R, COL_B, COL_G, COL_B_L
from chamber import CT

start_time = timeit.default_timer()
p = argparse.ArgumentParser(
    "Run SplineDist evaluation script using" + " point annotation data"
)
p.add_argument("dataPath", help="Path to folder containing eval images")
p.add_argument(
    "models",
    help="Glob-able path of SplineDist model(s) to check."
    ' If the glob matches include non-"pth" files, then the script will crash.',
)
p.add_argument(
    "--coco",
    help="path to COCO file of segmentation data for eval images."
    " Note: every image in this file must be inside the dataPath folder.",
)
p.add_argument(
    "--config",
    help="Path to the SplineDist models' config file"
    " (currently, models being checked must all have the same config). Default:"
    " configs/defaults.json",
    default="configs/defaults.json",
)
p.add_argument(
    "--legacy",
    action="store_true",
    help="Use legacy weight names (for loading old nets)",
)
p.add_argument(
    "--dest_dir",
    help="Folder in which to save error results, given as a path relative to the"
    " folder containing the network being evaluated (will be created if needed).",
)
p.add_argument(
    "--vis", action="store_true", help="Display visualization of the predicted eggs"
)
p.add_argument(
    "--vis_threshold",
    default=6,
    help="smallest absolute error at which" " to save error image examples",
    type=int,
)
p.add_argument(
    "--export_egg_json",
    action="store_true",
    help="whether to save"
    " a JSON file for each image, containing points that form the outline of every"
    " detected egg.",
)
p.add_argument(
    "--skip_err",
    action="store_true",
    help="count eggs in the images, skip the error calculations",
)
p.add_argument(
    "--no_dots",
    action="store_true",
    help="whether there are no point annotation files accompanying the images"
    ' (note: typically these match the filename of the image, but with "dots" added)',
)
p.add_argument(
    "--crop_test",
    action="store_true",
    help="if exporting egg JSON, save results for progressively"
    + " greater levels of cropping.",
)
opts = p.parse_args()


axis_norm = (0, 1)


def draw_line(img, point_lists, color=None):
    line_width = 1
    if color == None:
        color = [int(256 * i) for i in reversed(get_rand_color())]
    for line in point_lists:
        pts = np.array(line, dtype=np.int32)
        cv2.polylines(
            img, [pts], True, color, thickness=line_width, lineType=cv2.LINE_AA
        )


def prediction_renderer():
    _draw_polygons(coord, points, prob, show_dist=False)


def draw_polygons_based_on_overlap(
    polygons,
    annots,
    polygons_to_compare,
    img,
    flagged_color,
    draw_if_ok=True,
    debug=True,
    draw_iou_failures=True,
):
    iou_threshold = 0.50
    if debug:
        print("how many polygons to compare against?", len(polygons_to_compare))
    for i, polygon in enumerate(polygons):
        if debug:
            print("checking this polygon", polygon)
        centroid_match = False
        iou_match = False
        already_added = False
        index_pair = None
        for j, polygon_to_compare in enumerate(polygons_to_compare):
            if centroid_match:
                break
            if debug:
                print("against this one:", polygon_to_compare)
            if polygon.contains(polygon_to_compare.centroid):
                polygon_intersection = polygon.intersection(polygon_to_compare).area
                polygon_union = (
                    polygon.area + polygon_to_compare.area - polygon_intersection
                )
                if debug:
                    print(
                        "intersection over union:", polygon_intersection / polygon_union
                    )
                centroid_match = True
                if polygon_intersection / polygon_union > iou_threshold:
                    iou_match = True
                    if debug:
                        print("one contains the other.")
                    if draw_if_ok:
                        img_orig = np.array(img)
                        draw_line(img, [annots[i]], color=COL_W)
                        if debug:
                            cv2.imshow("before", img_orig)
                            cv2.imshow("after", img)
                            cv2.waitKey(0)
        if debug:
            print("found none that lined up. index pair is:", index_pair)
            print("centroid match?", centroid_match)
            print("iou match?", iou_match)
            print("already added?", already_added)
        if not centroid_match or (
            draw_iou_failures and centroid_match and not iou_match
        ):
            before_img = np.array(img)
            if not centroid_match:
                color = flagged_color
            elif draw_iou_failures and centroid_match and not iou_match:
                color = (255, 0, 255)
            draw_line(img, [annots[i]], color=color)
            if debug:
                cv2.imshow("before", before_img)
                cv2.imshow("after", img)
                cv2.waitKey(0)


def get_rand_color(pastel_factor=0.8):
    return [
        (x + pastel_factor) / (1.0 + pastel_factor)
        for x in [random.uniform(0, 1.0) for i in [1, 2, 3]]
    ]


# testImg = tifffile.imread(r"P:\Robert\splineDist\data\egg\images\Apr5_5left_38.tif")
# testJpg = np.array(Image.open(r"P:\Robert\objects_counting_dmap\egg_source\heldout\9_25_2020_img_0002_0_1_JK3J9.jpg"))

if opts.no_dots:
    X_names = []
    for single_path in opts.dataPath.split(","):
        X_names.extend(sorted(glob("%s/*.jpg" % single_path)))
elif opts.coco:
    coco_data = COCO(opts.coco)
    img_ids = list(coco_data.imgs.keys())
    X_names = []
    marked_pairs = set()
    for k in coco_data.imgs:
        img_path = f"{os.path.splitext(os.path.join(opts.dataPath,coco_data.imgs[k]['file_name']))[0]}.jpg"
        X_names.append(img_path)
else:
    Y_names = sorted(glob("%s/*_dots.png" % opts.dataPath))
    X_names = [el.replace("_dots.png", ".jpg") for el in Y_names]
X = list([np.array(img) for img in map(Image.open, X_names)])
if not opts.skip_err:
    if opts.coco:
        Y = []
        for i, img_id in enumerate(coco_data.imgs):
            loaded_img = cv2.imread(X_names[i])
            mask_img = Image.new("L", tuple(reversed(loaded_img.shape[0:2])), 0)
            annotations = coco_data.getAnnIds(imgIds=[img_id])
            random.shuffle(annotations)
            for i, ann in enumerate(annotations):
                ann = coco_data.anns[ann]
                ImageDraw.Draw(mask_img).polygon(
                    [int(el) for el in ann["segmentation"][0]],
                    outline=i + 1,
                    fill=i + 1,
                )
                if len(ann["segmentation"]) > 1:
                    for seg in ann["segmentation"][1:]:
                        ImageDraw.Draw(mask_img).polygon(
                            [int(el) for el in seg], outline=0, fill=0
                        )
            Y.append(np.array(mask_img))
    elif not opts.no_dots:
        Y = list([np.divide(np.array(img), 255) for img in map(Image.open, Y_names)])
tot_num_examples = len(X)
n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
# will the base 64 map from here match the earlier one?
# not needed in this case because we still have the original paths.
# print('testing an image:', X[0])
# cv2.imshow('debug', X[0])
# cv2.waitKey(0)
# print('testing a mask:', Y[0])
# print('number of eggs:', np.sum(Y[0]))
# input()
models = glob(opts.models)
config = Config(opts.config, n_channel)
phi = np.load(os.path.join(data_dir(), "phi_" + str(8) + ".npy"))


def get_interpolated_points(data, n_points=30):
    M = np.shape(data)[2]

    SplineContour = sg.SplineCurveVectorized(
        M, sg.B3(), True, np.transpose(data, [0, 2, 1]), useTorch=False
    )
    more_coords = SplineContour.sampleSequential(phi)
    # choose 30 points evenly divided by the range.
    sampling_interval = more_coords.shape[1] // n_points
    sampled_points = more_coords[:, slice(0, more_coords.shape[1], sampling_interval)]
    sampled_points = np.add(sampled_points, 1)
    sampled_points = sampled_points.astype(float).tolist()
    return sampled_points


def export_egg_json(img_basename, details, model_path, crop_level):
    if opts.dest_dir:
        out_folder = error_dir
    else:
        out_folder = Path(model_path).parent
    outfile = os.path.join(
        out_folder,
        f"{os.path.splitext(img_basename)[0]}_outlines"
        + (f"_crop{crop_level}px" if crop_level > 0 else "")
        + ".json",
    )
    egg_outlines = []
    sampled_points = get_interpolated_points(details["coord"])
    with open(outfile, "w") as f:
        json.dump(sampled_points, f, indent=4, ensure_ascii=False)


def create_and_save_img(img_gen_func):
    fig = plt.figure()
    plt.imshow(base_img, cmap="gray")
    img_gen_func()
    plt.gca().set_axis_off()
    current_figsize = fig.get_size_inches()
    fig.set_size_inches(current_figsize[0] * 2, current_figsize[1] * 2)
    plt.savefig("debug/temp.png", bbox_inches="tight", pad_inches=0, dpi=100)


saved_error_examples = {}

if opts.export_egg_json and opts.crop_test:
    crop_levels = range(32)
else:
    crop_levels = range(1)

for model_path in models:
    if opts.dest_dir:
        model_parent = Path(model_path).parent
        error_dir = os.path.join(model_parent, opts.dest_dir)
        Path(error_dir).mkdir(parents=True, exist_ok=True)
    else:
        model_dir = Path(model_path).parent
        error_dir = os.path.join(model_dir, "error_examples")
    true_values = []
    predicted_values = []
    errors_by_img = {}
    model = SplineDist2D(config)
    model.cuda()
    model.train(False)
    loaded_model = torch.load(model_path)
    if opts.legacy:
        state_dict_new = copy.deepcopy(loaded_model)
        for key in loaded_model:
            if "unet" in key:
                state_dict_new[key.replace("unet", "backbone")] = state_dict_new.pop(
                    key
                )
        model.load_state_dict(state_dict_new)
    else:
        model.load_state_dict(loaded_model)
    for i, img in enumerate(X):
        predict_start = timeit.default_timer()
        img_basename = os.path.basename(X_names[i])
        img_orig = img
        predict_resize_factor = 1.0
        display_resize_factor = 1.0
        img = cv2.resize(
            img, (0, 0), fx=predict_resize_factor, fy=predict_resize_factor
        )
        img_display = cv2.resize(
            img_orig, (0, 0), fx=display_resize_factor, fy=display_resize_factor
        )
        img = normalize(img, 1, 99.8, axis=axis_norm)
        img = img.astype(np.float32)
        labels, details = model.predict_instances(img)
        num_predicted = len(details["points"])
        print("img %i of %i" % (i, tot_num_examples))
        print("num predicted:", num_predicted)
        if not opts.skip_err:
            chamber_type_path = os.path.join(
                Path(X_names[i]).parent, "chamber_types.json"
            )
            has_chamber_type = os.path.exists(chamber_type_path)
            if has_chamber_type:
                pass
                with open(chamber_type_path) as f:
                    ct_map = json.load(f)
                    ct_instance = CT[ct_map[img_basename]].value()
                    num_labeled = ct_instance.numRows * ct_instance.numCols
                    if type(ct_instance) == type(CT.large.value()):
                        num_labeled *= 4
            elif opts.coco:
                num_labeled = len(np.unique(Y[i])) - 1
            else:
                num_labeled = np.sum(Y[i])
            print("num labeled:", num_labeled)
            path_components = os.path.normpath(X_names[i]).split(os.sep)
            abs_err = abs(num_labeled - num_predicted)
            errors_by_img[f"{path_components[-2]}_{path_components[-1]}"] = abs_err
            true_values.append(num_labeled)
        predicted_values.append(num_predicted)
        if opts.export_egg_json:
            for crop_level in crop_levels:
                crop_slice = slice(0, -crop_level if crop_level else None)
                img = img_orig[crop_slice, crop_slice]
                img = normalize(img, 1, 99.8, axis=axis_norm)
                labels, details = model.predict_instances(img)
                export_egg_json(img_basename, details, model_path, crop_level)

        if opts.vis:
            img_show = cv2.cvtColor(np.array(img_display), cv2.COLOR_RGB2BGR)
            coord, points, prob = details["coord"], details["points"], details["prob"]
            for egg_contour in coord:
                annot = get_interpolated_points(
                    np.expand_dims(
                        np.array(
                            [
                                tuple(
                                    reversed(
                                        [
                                            int(
                                                el
                                                * (
                                                    display_resize_factor
                                                    / predict_resize_factor
                                                )
                                            )
                                            for el in tup
                                        ]
                                    )
                                )
                                for tup in zip(*egg_contour)
                            ]
                        ).T,
                        0,
                    )
                )[0]
                draw_line(img_show, [annot])
            cv2.imwrite(
                os.path.join(
                    "debug",
                    f"errors_{path_components[-2]}_{path_components[-1]}_{os.path.basename(model_path)}.png",
                ),
                img_show,
            )
        if (
            not opts.skip_err
            and not opts.no_dots
            and opts.vis
            and abs_err >= opts.vis_threshold
            and num_labeled >= 11
            and num_labeled <= 40
        ):
            # should I add to this code path?
            # three of the conditions above seem to still apply here:
            # 1) calculating error; 2) want visuals, 3) abs_err > some threshold.
            # the final condition appears too specific for the new case,
            # because there will be multiple exported images for each
            # basename.
            # img_show = img if img.ndim == 2 else img[..., 0]
            if (
                not opts.coco
                and img_basename in saved_error_examples
                and saved_error_examples[img_basename]["abs_err"] < abs_err
            ):
                continue

            if opts.dest_dir:
                out_folder = error_dir
            else:
                out_folder = Path(model_path).parent
            outfile = os.path.join(
                error_dir, f"{os.path.splitext(img_basename)}_egg_outlines.json"
            )
            print("trying to make the following directory:", error_dir)
            Path(error_dir).mkdir(parents=True, exist_ok=True)
            gt_pts = np.where(Y[i] > 0)
            # fig = plt.figure()
            # img_show = img if img.ndim == 2 else img[..., 0]
            coord, points, prob = details["coord"], details["points"], details["prob"]
            # plt.imshow(img_show, cmap="gray")
            # _draw_polygons(coord, points, prob, show_dist=True)
            for fname in (
                "temp.png",
                "debug/temp.png",
            ):
                try:
                    os.unlink(fname)
                except FileNotFoundError:
                    pass

            # create_and_save_img(prediction_renderer)
            # predictions = cv2.imread("debug/temp.png")
            # plt.close("all")
            # base_img = cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR)
            base_img = np.array(img_orig)

            def gt_renderer():
                plt.scatter(gt_pts[1], gt_pts[0], 4, [[1, 0, 0.157]])

            if opts.coco:
                resize_factor = 2.5
                gt_polygons = []
                gt_annots = []
                for annot in coco_data.imgToAnns[img_ids[i]]:
                    annot = annot["segmentation"][0]
                    annot = list(zip(annot[::2], annot[1::2]))
                    annot = [
                        tuple([int(el * resize_factor) for el in tup]) for tup in annot
                    ]
                    gt_annots.append(annot)
                    annot_as_array = np.array(annot)
                    gt_polygons.append(Polygon(annot_as_array))
                    # cv2.drawMarker(
                    # ground_truth,
                    # tuple([round(el) for el in gt_centroids[-1].coords[0]]),
                    # (255, 0, 0),
                    # markerType=cv2.MARKER_STAR,
                    # )
                    # draw_line(ground_truth, [gt_annots[-1]])
                base_img = cv2.resize(
                    cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR),
                    (0, 0),
                    fx=resize_factor,
                    fy=resize_factor,
                )
                predictions = np.zeros(base_img.shape, dtype=np.uint8)
                ground_truth = np.zeros(base_img.shape, dtype=np.uint8)
                predicted_polygons = []
                predicted_annots = []
                for egg_contour in coord:
                    annot = get_interpolated_points(
                        np.expand_dims(
                            np.array(
                                [
                                    tuple(
                                        reversed(
                                            [int(el * resize_factor) for el in tup]
                                        )
                                    )
                                    for tup in zip(*egg_contour)
                                ]
                            ).T,
                            0,
                        )
                    )[0]
                    predicted_annots.append(annot)
                    annot_as_array = np.array(annot)
                    # print('annot?', annot_as_array)
                    predicted_polygons.append(Polygon(annot_as_array))
                    draw_line(predictions, [annot])
                    # cv2.drawMarker(
                    #     predictions,
                    #     tuple([round(el) for el in predicted_centroids[-1].coords[0]]),
                    #     (255, 0, 0),
                    #     markerType=cv2.MARKER_STAR,
                    # )
                # for i, polygon in enumerate(gt_polygons):
                #     for polygon_to_compare in predicted_polygons:
                #         if polygon.contains(polygon_to_compare.centroid):
                #             draw_line(ground_truth, [gt_annots[i]], color=COL_W)
                #             break
                #         draw_line(ground_truth, [gt_annots[i]], color=COL_R)
                draw_polygons_based_on_overlap(
                    gt_polygons,
                    gt_annots,
                    predicted_polygons,
                    ground_truth,
                    COL_G,
                    debug=False,
                )
                draw_polygons_based_on_overlap(
                    predicted_polygons,
                    predicted_annots,
                    gt_polygons,
                    ground_truth,
                    COL_R,
                    draw_if_ok=False,
                    debug=False,
                    draw_iou_failures=False,
                )
                ground_truth = cv2.addWeighted(
                    base_img,
                    1,
                    ground_truth.astype(np.uint8),
                    0.4,
                    0,
                )
                predictions = cv2.addWeighted(
                    base_img, 1, predictions.astype(np.uint8), 0.4, 0
                )
                # (0, 0),
                # fx = predictions.shape[1]
                # )
                # cv2.imshow("original image:", base_img)
                # cv2.imshow("combined", ground_truth)
                # cv2.imshow("pred", predictions)
                # cv2.waitKey(0)
            else:
                create_and_save_img(gt_renderer)
                ground_truth = cv2.imread("debug/temp.png")
                predictions = img_show
            # cv2.imshow("gt", ground_truth)
            # cv2.waitKey(0)

            if (
                ground_truth.shape[0] >= ground_truth.shape[1]
            ):  # the images are taller than they are wide
                print("original base image shape:", base_img.shape)
                print(
                    "rescaling factor for x:", ground_truth.shape[1] / base_img.shape[1]
                )
                print("and for y:", ground_truth.shape[0] / base_img.shape[0])
                base_img = cv2.cvtColor(
                    cv2.resize(
                        base_img,
                        (0, 0),
                        fx=ground_truth.shape[1] / base_img.shape[1],
                        fy=ground_truth.shape[0] / base_img.shape[0],
                    ),
                    cv2.COLOR_RGB2BGR,
                )
                predictions = cv2.resize(
                    predictions,
                    (0, 0),
                    fx=ground_truth.shape[1] / predictions.shape[1],
                    fy=ground_truth.shape[0] / predictions.shape[0],
                )
                combined_img = np.zeros(
                    (
                        max(ground_truth.shape[0], predictions.shape[0]),
                        ground_truth.shape[1] + predictions.shape[1] * 2 + 2 * 10,
                        3,
                    )
                )
                print("base image after resize:", base_img.shape)
                combined_img[:, : ground_truth.shape[1]] = ground_truth
                print("combined image shape:", combined_img.shape)
                print("predictions shape:", predictions.shape)
                print("ground truth shape:", ground_truth.shape)
                print("base img shape:", base_img.shape)
                combined_img[
                    :, ground_truth.shape[1] + 10 : 2 * ground_truth.shape[1] + 10
                ] = base_img
                combined_img[:, 2 * ground_truth.shape[1] + 20 :] = predictions
            else:
                base_img = cv2.cvtColor(
                    cv2.resize(
                        base_img,
                        (0, 0),
                        fx=ground_truth.shape[1] / base_img.shape[1],
                        fy=ground_truth.shape[0] / base_img.shape[0],
                    ),
                    cv2.COLOR_RGB2BGR,
                )
                predictions = cv2.resize(
                    predictions,
                    (0, 0),
                    fx=ground_truth.shape[1] / predictions.shape[1],
                    fy=ground_truth.shape[0] / predictions.shape[0],
                )
                combined_img = np.zeros(
                    (
                        ground_truth.shape[0] + predictions.shape[0] * 2 + 2 * 10,
                        max(ground_truth.shape[1], predictions.shape[1]),
                        3,
                    )
                )
                combined_img[: ground_truth.shape[0], :] = ground_truth
                combined_img[
                    ground_truth.shape[0] + 10 : 2 * ground_truth.shape[0] + 10, :
                ] = base_img
                combined_img[2 * ground_truth.shape[0] + 20 :, :] = predictions
            # save the combined image with a filename containing the
            # original filename AND the number of predicted and labeled eggs.
            # also need to save the name of the model that was run...
            # will all that info make the filename too long?
            # what alternative is there?
            # perhaps just to save one representative error example per image being checked.
            # we keep a mapping of image name to level of error, and if the previous was smaller
            # than the current error, then we overwrite. This seems to make sense
            # as far as keeping the storage footprint smaller.
            # think this is now implemented...
            if img_basename in saved_error_examples:
                os.unlink(saved_error_examples[img_basename]["path"])
            save_path = os.path.join(
                error_dir,
                "%iabs_%s_%s_%ipredicted_%ilabeled.png"
                % (
                    abs_err,
                    img_basename.split(".jpg")[0],
                    os.path.basename(model_path),
                    num_predicted,
                    num_labeled,
                ),
            )
            saved_error_examples[os.path.basename(X_names[i])] = {
                "path": save_path,
                "abs_err": abs_err,
            }
            cv2.imwrite(save_path, combined_img)

        print("time per prediction:", timeit.default_timer() - predict_start)
    # if it's running in batch mode, need to save the final error stats
    # for each model. Keep them in the same JSON file.
    if not opts.skip_err:
        true_values = np.array(true_values)
        predicted_values = np.array(predicted_values)
        abs_diff = np.abs(np.subtract(predicted_values, true_values))
        abs_rel_errors = np.divide(abs_diff, true_values)
        mean_abs_error = np.mean(abs_diff)
        max_abs_error = int(np.max(abs_diff))
        mean_rel_error = np.mean(
            abs_rel_errors[(abs_rel_errors != np.infty) & (~np.isnan(abs_rel_errors))]
        )
        mean_rel_error_0to10 = np.mean(
            abs_rel_errors[
                (abs_rel_errors != np.infty)
                & (~np.isnan(abs_rel_errors))
                & (true_values < 11)
            ]
        )
        mean_rel_error_11to40 = np.mean(
            abs_rel_errors[
                (abs_rel_errors != np.infty)
                & ((true_values >= 11) & (true_values < 41))
            ]
        )
        mean_rel_error_41plus = np.mean(
            abs_rel_errors[(abs_rel_errors != np.infty) & (true_values >= 41)]
        )
        if opts.dest_dir:
            dest = os.path.join(
                error_dir, f"{os.path.basename(model_path)}_errors.json"
            )
        else:
            dest = "%s_errors.json" % model_path
        with open(dest, "w") as f:
            json.dump(
                {
                    "mean_abs_error": mean_abs_error,
                    "max_abs_error": max_abs_error,
                    "mean_rel_error": mean_rel_error,
                    "mean_rel_error_0to10": mean_rel_error_0to10,
                    "mean_rel_error_11to40": mean_rel_error_11to40,
                    "mean_rel_error_41plus": mean_rel_error_41plus,
                    "errors_by_img": errors_by_img,
                },
                f,
                ensure_ascii=False,
                indent=4,
            )
print("Total run time:", timeit.default_timer() - start_time)
