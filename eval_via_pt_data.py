# Run the evaluation script without requiring instance segmentation for the
# input data

import argparse
import copy
from csbdeep.utils import normalize
from glob import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
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
p.add_argument('--legacy', action='store_true', help='Use legacy weight names (for loading old nets)')
p.add_argument('--vis', action='store_true', help='Display visualization of the predicted eggs')
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
            if 'unet' in key:
                state_dict_new[key.replace('unet', 'backbone')] = state_dict_new.pop(key)
        model.load_state_dict(state_dict_new)
    else:
        model.load_state_dict(loaded_model)
    for i, img in enumerate(X):
        predict_start = timeit.default_timer()
        img = normalize(img, 1, 99.8, axis=axis_norm)
        labels, details = model.predict_instances(img)
        num_predicted = len(details["points"])
        num_labeled = np.sum(Y[i])
        print("img %i of %i" % (i, tot_num_examples))
        print("num predicted:", num_predicted)
        print("num labeled:", num_labeled)
        errors_by_img[os.path.basename(X_names[i])] = abs(num_labeled - num_predicted)
        predicted_values.append(num_predicted)
        true_values.append(num_labeled)
        if opts.vis:
            plt.figure(figsize=(13, 10))
            img_show = img if img.ndim == 2 else img[..., 0]
            coord, points, prob = details["coord"], details["points"], details["prob"]
            plt.subplot(121)
            plt.imshow(img_show, cmap="gray")
            plt.axis("off")
            a = plt.axis()
            _draw_polygons(coord, points, prob, show_dist=True)
            plt.axis(a)
            # plt.subplot(122)
            # plt.imshow(img_show, cmap="gray")
            # plt.axis("off")
            # plt.imshow(lbl, cmap=lbl_cmap, alpha=0.5)
            plt.tight_layout()
            plt.show()
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
    with open("%s_errors.json" % model_path, "w") as f:
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
