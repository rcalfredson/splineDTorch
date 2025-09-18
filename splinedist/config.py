import json
from splinedist.models.backbone_types import BackboneTypes
from splinedist.utils import normalize_grid

DEFAULT_CONFIG = "configs/defaults.json"


class Config:
    def __init__(self, config_file, n_channel_in=3, contoursize_max=None):
        with open(config_file, "r") as my_f:
            self.conf_from_file = json.load(my_f)
        with open(DEFAULT_CONFIG, "r") as my_f:
            self.default_conf = json.load(my_f)
        self.axes = self.set_conf_param("axes")
        self.backbone = BackboneTypes[self.set_conf_param("backbone").lower()]
        self.contoursize_max = (
            contoursize_max if contoursize_max is None else int(contoursize_max)
        )
        self.deform_sigma = float(self.set_conf_param("deform_sigma"))
        self.expanded_batch_norm = self.set_conf_param("expanded_batch_norm")
        self.focused_patch_proportion = self.set_conf_param("focused_patch_proportion")
        self.force_even_patch_size = self.set_conf_param("force_even_patch_size")
        self.grid = normalize_grid(self.set_conf_param("grid_subsampling_factor"), 2)
        self.lr_reduct_factor = self.set_conf_param("lr_reduct_factor")
        self.lr_patience = self.set_conf_param("lr_patience")
        self.n_channel_in = n_channel_in
        self.n_dim = int(self.set_conf_param("n_dim"))
        self.sample_patches = self.set_conf_param("sample_patches")
        self.skip_empties = self.set_conf_param("skip_empties")
        self.skip_partials = self.set_conf_param("skip_partials")
        self.n_params = 2 * int(self.set_conf_param("n_control_points"))
        self.train_background_reg = int(self.set_conf_param("train_background_reg"))
        self.train_batch_size = int(self.set_conf_param("train_batch_size"))
        self.train_completion_crop = int(self.set_conf_param("train_completion_crop"))
        self.train_dist_loss = self.set_conf_param("train_dist_loss")
        self.train_epochs = self.set_conf_param("train_epochs")
        self.train_foreground_only = self.set_conf_param("train_foreground_only")
        self.train_learning_rate = float(self.set_conf_param("train_learning_rate"))
        self.train_loss_weights = tuple(self.set_conf_param("train_loss_weights"))
        self.train_patch_size = tuple(self.set_conf_param("train_patch_size"))
        self.train_shape_completion = self.set_conf_param("train_shape_completion")
        self.train_steps_per_epoch = self.set_conf_param("train_steps_per_epoch")
        self.validation_batch_size = self.set_conf_param("validation_batch_size")
        self.validation_steps_per_epoch = self.set_conf_param(
            "validation_steps_per_epoch"
        )
        self.zoom_min = self.set_conf_param("zoom_min")
        self.zoom_max = self.set_conf_param("zoom_max")

        # default config (can be overwritten by kwargs below)
        if self.backbone in (BackboneTypes.unet_full, BackboneTypes.unet_reduced):
            self.n_depth = 3
            self.kernel_size = 3, 3
            self.n_filter_base = 32
            self.n_conv_per_depth = 2
            self.pool = 2, 2
            self.activation = "relu"
            self.last_activation = "relu"
            self.net_conv_after_backbone = 128
        elif self.backbone == BackboneTypes.fcrn_a:
            self.n_depth = 3
            self.kernel_size = 3, 3
            self.n_filter_base = 32
            self.n_conv_per_depth = 2
            self.pool = 2, 2
            self.activation = "relu"
            self.last_activation = "relu"
            self.net_conv_after_backbone = 128

    def set_conf_param(self, param_name):
        val = (
            self.conf_from_file[param_name]
            if param_name in self.conf_from_file
            else self.default_conf[param_name]
        )
        setattr(self, param_name, val)
        return getattr(self, param_name)
