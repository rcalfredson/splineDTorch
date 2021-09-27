import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch

sys.path.append(os.path.abspath("./"))
from splinedist.config import Config
from splinedist.models.model2d import SplineDist2D

p = argparse.ArgumentParser(description="measure the weights of an existing neural net")
p.add_argument("model_path", help="path to the model whose weights to measure")
p.add_argument(
    "--config", help="path to the model's config file", default="configs/defaults.json"
)

opts = p.parse_args()
N_CHANNELS = 3
config = Config(opts.config, N_CHANNELS)
model = SplineDist2D(config)
model.cuda()
model.train(False)
model.load_state_dict(torch.load(opts.model_path))
weights_by_layer = {}
print("loaded model?", model)
for param in model.named_parameters():
    print("param?", param)
    layer_name = ".".join(param[0].split(".")[:2])
    weights_flattened = param[1].view(-1)
    print("layer name?", layer_name)
    # print("all params in flattened form?", param[1].view(-1))
    if layer_name not in weights_by_layer:
        weights_by_layer[layer_name] = []
        weights_by_layer[layer_name].append(weights_flattened)
    else:
        weights_by_layer[layer_name].append(weights_flattened)
    # print("weights by layer?", weights_by_layer)
for layer in weights_by_layer:
    weights_by_layer[layer] = torch.cat(weights_by_layer[layer])
print("final weights by layer:", weights_by_layer)
stats_by_layer = {layer_name: {} for layer_name in weights_by_layer.keys()}
for layer in weights_by_layer:
    plt.figure()
    weights_as_np = weights_by_layer[layer].cpu().detach().numpy()
    plt.hist(weights_as_np, bins=25)
    plt.title(f"Weights for layer {layer}")
    print('min and max:', np.amin(weights_as_np), np.amax(weights_as_np))
    plt.show()
    stats_by_layer[layer] = {
        "mean": torch.mean(weights_by_layer[layer]),
        "var": torch.var(weights_by_layer[layer], unbiased=False),
    }
print('final stats by layer:', stats_by_layer)
for layer in stats_by_layer:
    print('Layer name:', layer)
    print(f"Mean: {stats_by_layer[layer]['mean'].item()}")
    print(f"Variance: {stats_by_layer[layer]['var'].item()}")
    print()