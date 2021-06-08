# Run the evaluation script without requiring instance segmentation for the
# input data

import argparse
import copy
from csbdeep.utils import normalize
import cv2
from glob import glob
import json
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from PIL import Image
from splinedist.config import Config
from splinedist.models.model2d import SplineDist2D
from splinedist.plot.plot import _draw_polygons
import timeit
import torch


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
opts = p.parse_args()


axis_norm = (0, 1)

# testImg = tifffile.imread(r"P:\Robert\splineDist\data\egg\images\Apr5_5left_38.tif")
# testJpg = np.array(Image.open(r"P:\Robert\objects_counting_dmap\egg_source\heldout\9_25_2020_img_0002_0_1_JK3J9.jpg"))

Y_names = sorted(glob("%s/*_dots.png" % opts.dataPath))
X_names = [el.replace("_dots.png", ".jpg") for el in Y_names]
X = list([np.array(img) for img in map(Image.open, X_names)])
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


def create_and_save_img(img_gen_func):
    fig = plt.figure()
    plt.imshow(img_show, cmap="gray")
    img_gen_func()
    plt.gca().set_axis_off()
    current_figsize = fig.get_size_inches()
    fig.set_size_inches(current_figsize[0] * 2, current_figsize[1] * 2)
    plt.savefig("debug/temp.png", bbox_inches="tight", pad_inches=0, dpi=100)


saved_error_examples = {}

for model_path in models:
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
        # img = normalize(img, 1, 99.8, axis=axis_norm)
        img = img.astype(np.float32)
        labels, details = model.predict_instances(img)
        num_predicted = len(details["points"])
        num_labeled = np.sum(Y[i])
        print("img %i of %i" % (i, tot_num_examples))
        print("num predicted:", num_predicted)
        print("num labeled:", num_labeled)
        abs_err = abs(num_labeled - num_predicted)
        errors_by_img[os.path.basename(X_names[i])] = abs_err
        predicted_values.append(num_predicted)
        true_values.append(num_labeled)
        if (
            opts.vis
            and abs_err > 6
            and (
                img_basename not in saved_error_examples
                or saved_error_examples[img_basename]["abs_err"] < abs_err
            )
        ):
            # create parent dir for error examples
            model_dir = Path(model_path).parent
            error_dir = os.path.join(model_dir, "error_examples")
            Path(error_dir).mkdir(parents=True, exist_ok=True)
            gt_pts = np.where(Y[i] > 0)
            # fig = plt.figure()
            img_show = img if img.ndim == 2 else img[..., 0]
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

            def prediction_renderer():
                _draw_polygons(coord, points, prob, show_dist=True)

            create_and_save_img(prediction_renderer)
            predictions = cv2.imread("debug/temp.png")
            plt.close("all")

            def gt_renderer():
                plt.scatter(gt_pts[1], gt_pts[0], 4, [[1, 0, 0.157]])

            create_and_save_img(gt_renderer)
            ground_truth = cv2.imread("debug/temp.png")
            # cv2.imshow("pred", predictions)
            # cv2.imshow("gt", ground_truth)
            # cv2.waitKey(0)

            if (
                ground_truth.shape[0] >= ground_truth.shape[1]
            ):  # the images are taller than they are wide
                combined_img = np.zeros(
                    (
                        max(ground_truth.shape[0], predictions.shape[0]),
                        ground_truth.shape[1] + predictions.shape[1] + 10,
                        3,
                    )
                )
                combined_img[:, : ground_truth.shape[1]] = ground_truth
                combined_img[:, ground_truth.shape[1] + 10 :] = predictions
            else:
                combined_img = np.zeros(
                    (
                        ground_truth.shape[0] + predictions.shape[0] + 10,
                        max(ground_truth.shape[1], predictions.shape[1]),
                        3,
                    )
                )
                combined_img[: ground_truth.shape[0], :] = ground_truth
                combined_img[ground_truth.shape[0] + 10 :, :] = predictions
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
                "%s_%iabs_%s_%ipredicted_%ilabeled.png"
                % (
                    img_basename.split(".jpg")[0],
                    abs_err,
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
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)
    abs_diff = np.abs(np.subtract(predicted_values, true_values))
    abs_rel_errors = np.divide(abs_diff, true_values)
    mean_abs_error = np.mean(abs_diff)
    max_abs_error = np.max(abs_diff)
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
            (abs_rel_errors != np.infty) & ((true_values >= 11) & (true_values < 41))
        ]
    )
    mean_rel_error_41plus = np.mean(
        abs_rel_errors[(abs_rel_errors != np.infty) & (true_values >= 41)]
    )
    if opts.dest_dir:
        model_parent = Path(model_path).parent
        error_dir = os.path.join(model_parent, opts.dest_dir)
        Path(error_dir).mkdir(parents=True, exist_ok=True)
        dest = os.path.join(error_dir,
            f"{os.path.basename(model_path)}_errors.json")
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
