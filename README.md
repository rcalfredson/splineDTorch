# splineDTorch

**SplineDist object detection implemented in PyTorch.**  
This project adapts the [SplineDist architecture](https://github.com/uhlmanngroup/splinedist) (a cubic-splines version of StarDist) into a PyTorch framework.

---

## Overview

SplineDist is an object detection model that predicts object outlines as parametric splines.  
This PyTorch port supports **single-category detection** only: the network is trained to detect instances of one object type at a time (e.g., wheels, nuclei, cells).

---

## Dataset Preparation

The training setup expects a directory structure like:

```
<base_data_dir>/
    <object_name>/
        images/
            img_001.png
            img_002.tif
            ...
```

- Each object type you want to detect has its own directory.  
- Within that directory, place all training images inside an `images/` subfolder.  
- Annotations should be prepared with COCO-style segmentation. We recommend [coco-annotator](https://github.com/jsbroks/coco-annotator).  
- Export a **COCO JSON file** that contains your ground-truth masks.

Masks are dynamically generated at training time from this COCO file.

---

## Configuration

Training relies on a JSON config file that specifies architecture and training hyperparameters.  

- Default values live in `configs/defaults.json`.  
- You can override them with your own config (example: `configs/wheel-detection-net-config.json`).  
- Key parameters include:
  - **backbone** (`unet_full`, `unet_reduced`, `fcrn_a`)  
  - **n_control_points** (number of spline control points)  
  - **train_patch_size**, **train_batch_size**, **train_epochs**, **train_learning_rate**  
  - **data augmentation settings** (zoom, deformation, patch sampling)  
  - **loss weighting** and **regularization flags**  

See [`splinedist/config.py`](splinedist/config.py) for the full list of supported keys.

---

## Training

To train a model:

```
python train.py "wheel_detection" \
    --data_base_dir /home/tracking/ml_data \
    --coco_file_path "/home/tracking/ml_data/wheel_detection/wheel-orientation-1.json" \
    --config configs/wheel-detection-net-config.json
```

Arguments:
- `data_path` (positional): dataset subfolder (relative to `--data_base_dir`)  
- `--data_base_dir`: base path for datasets  
- `--coco_file_path`: path to COCO annotations file  
- `--config`: path to JSON config  
- `--model_path`: resume training from an existing model (optional)  
- `--plot`: enable live training plots  
- `--val_interval`: validation frequency (epochs)  

Models and logs are saved under `results_<hostname>/`.

---

## Inference

Once trained, you can run predictions like:

```
# Run the model
_, predictions = model.predict_instances(image)

# Example: extract circle centers from outlines
centroids = []
if "outlines" in predictions:
    for outline in predictions["outlines"]:
        outline_arr = np.array(outline)
        cx, cy, r = fit_circle_kasa(outline_arr)
        centroids.append((cy, cx))
```

> The circle-fitting step is optional, shown here only for a circular-object use case.

---

## Dependencies

- Python 3.9+ (recommended: 3.13)
- PyTorch  
- NumPy, SciPy, Matplotlib  
- Pillow  
- [pycocotools](https://github.com/cocodataset/cocoapi)  
- tifffile  
- tqdm

---

## Credits

- Original SplineDist: [Uhlmann Group @ EMBL-EBI](https://github.com/uhlmanngroup/splinedist)  
