from typing import Optional
import matplotlib
import numpy as np
from splinedist.config import Config
from splinedist.models.database import SplineDistData2D, SplineDistDataStatic
from splinedist.models.model2d import SplineDist2D
import torch

import timeit


class Looper:
    def __init__(
        self,
        network: SplineDist2D,
        config: Config,
        device,
        loss,
        optimizer,
        augmenter,
        X,
        Y,
        validation: bool,
        plots: Optional[matplotlib.axes.Axes] = None,
        left_col_plots: str = None,
    ):
        self.network = network
        self.config = config
        self.device = device
        self.loss = loss
        self.optimizer = optimizer
        self.augmenter = augmenter
        self.X, self.Y = X, Y
        self.dataset_size = len(X)
        self.validation = validation
        self.plots = plots
        self.step_counter_to_reset_each_dataset_pass = 0
        if plots is not None:
            assert (
                left_col_plots is not None
            ), "left_col_plots must have a value if plots are set"
        self.left_col_plots = left_col_plots

        self.epochs = self.config.train_epochs
        # when should validation prevent duplicates from getting in?

        self.data_kwargs = dict(
            n_params=self.config.n_params,
            patch_size=self.config.train_patch_size,
            grid=self.config.grid,
            shape_completion=self.config.train_shape_completion,
            b=self.config.train_completion_crop,
            foreground_prob=self.config.train_foreground_only,
            contoursize_max=self.config.contoursize_max,
        )

        self.n_samples_per_img = 20 if self.validation else 3
        self.n_samples_per_epoch = self.dataset_size * self.n_samples_per_img

        if not self.validation:
            if self.config.train_steps_per_epoch is not None:
                self.steps_per_epoch = self.config.train_steps_per_epoch
                self.n_samples_per_epoch = self.config.train_steps_per_epoch * self.config.train_batch_size
            else:
                self.steps_per_epoch = (
                    self.n_samples_per_epoch / self.config.train_batch_size
                )
            self.data = SplineDistData2D(
                self.X,
                self.Y,
                self.config.train_batch_size,
                length=self.steps_per_epoch,
                n_samples=self.n_samples_per_img,
                augmenter=self.augmenter,
                **self.data_kwargs,
            )
        else:
            if self.config.validation_steps_per_epoch is not None:
                self.steps_per_epoch = self.config.validation_steps_per_epoch
                self.n_samples_per_epoch = self.config.validation_steps_per_epoch * self.config.validation_batch_size
            else:
                self.steps_per_epoch = (
                    self.n_samples_per_epoch / self.config.validation_batch_size
                )
            self.setup_data_for_validation()
        self.running_loss = []
        self.running_mean_abs_err = []

    def setup_data_for_validation(self):
        _data_val = SplineDistData2D(
            self.X,
            self.Y,
            batch_size=self.n_samples_per_epoch,
            length=1,
            n_samples=self.n_samples_per_img,
            skip_dist_prob_calc=True,
            augmenter=self.augmenter,
            **self.data_kwargs,
        )
        data_val_source = _data_val[0]
        self.data = SplineDistDataStatic(
            data_val_source[0],
            data_val_source[1],
            batch_size=self.config.validation_batch_size,
            length=self.steps_per_epoch,
            **self.data_kwargs,
        )

    def run(self, i):
        self.err = []
        self.abs_err = []
        self.true_values = []
        self.predicted_values = []
        self.running_loss.append(0)
        self.network.train(not self.validation)
        for j, datum in enumerate(self.data):
            # start_t = timeit.default_timer()
            patches = torch.from_numpy(datum[0][0]).float().to(self.device)
            patches = patches.permute(0, 3, 1, 2)
            true_prob = torch.from_numpy(datum[1][0]).float().to(self.device)
            true_dist = torch.from_numpy(datum[1][1]).float().to(self.device)
            if not self.validation:
                self.optimizer.zero_grad()
            result = self.network(patches)
            pred_prob = result[0].permute(0, 2, 3, 1)
            pred_dist = result[1].permute(0, 2, 3, 1)
            loss = self.network.loss([true_prob, true_dist], [pred_prob, pred_dist])
            if not self.validation:
                loss.backward()
                self.optimizer.step()
                self.network.train(False)
            # start_t = timeit.default_timer()
            for patch_i in range(patches.shape[0]):
                _, predict_result = self.network.predict_instances(
                    patches[patch_i].permute(1, 2, 0).cpu()
                )
                self.err.append(datum[2][patch_i] - len(predict_result["points"]))
                self.abs_err.append(abs(self.err[-1]))
                self.true_values.append(datum[2][patch_i])
                self.predicted_values.append(len(predict_result["points"]))
            # print("total predict time:", timeit.default_timer() - start_t)

            if not self.validation:
                self.network.train(True)
            self.running_loss[-1] += (
                patches.shape[0] * loss.item() / (self.n_samples_per_epoch)
            )
            self.step_counter_to_reset_each_dataset_pass += 1
            if self.step_counter_to_reset_each_dataset_pass == self.dataset_size:
                self.data.reset_index_map()
                self.step_counter_to_reset_each_dataset_pass = 0

            if j + 1 == self.steps_per_epoch:
                break
            # end_of_block = timeit.default_timer()
            # if 'old_end_of_block' in locals():
            #     time_diff = end_of_block - old_end_of_block
            #     old_end_of_block = end_of_block
            # else:
            #     time_diff = end_of_block - start_t
            #     old_end_of_block = start_t
            # print(f'time needed for one step: {time_diff:.3f}')
        self.update_errors()
        if self.plots is not None:
            self.plot()
        self.log()
        return self.mean_abs_err

    def update_errors(self):
        self.mean_err = sum(self.err) / self.n_samples_per_epoch
        self.mean_abs_err = sum(self.abs_err) / self.n_samples_per_epoch
        self.running_mean_abs_err.append(self.mean_abs_err)
        self.std = np.array(self.err).std()

    def plot(self):
        """Plot true vs predicted counts and loss."""
        # true vs predicted counts
        true_line = [[0, max(self.true_values)]] * 2  # y = x
        epochs = np.arange(1, len(self.running_loss) + 1)
        self.plots[0].cla()
        self.plots[0].set_title("Train" if not self.validation else "Valid")

        if self.left_col_plots == "scatter":
            self.plots[0].set_xlabel("True value")
            self.plots[0].set_ylabel("Predicted value")
            self.plots[0].plot(*true_line, "r-")
            self.plots[0].scatter(self.true_values, self.predicted_values)
        elif self.left_col_plots == "mae":
            self.plots[0].set_xlabel("Epoch")
            self.plots[0].set_ylabel("Mean absolute error")
            self.plots[0].set_ylim((0, 4))
            self.plots[0].plot(epochs, self.running_mean_abs_err)

        # loss
        self.plots[1].cla()
        self.plots[1].set_title("Train" if not self.validation else "Valid")
        self.plots[1].set_xlabel("Epoch")
        self.plots[1].set_ylabel("Loss")
        self.plots[1].set_ylim((0, 0.5))
        self.plots[1].plot(epochs, self.running_loss)

        matplotlib.pyplot.pause(0.01)
        matplotlib.pyplot.tight_layout()

    def log(self):
        print(
            f"{'Train' if not self.validation else 'Valid'}:\n"
            f"\tAverage loss: {self.running_loss[-1]:3.4f}\n"
            f"\tMean error: {self.mean_err:3.3f}\n"
            f"\tMean absolute error: {self.mean_abs_err:3.3f}\n"
            f"\tError deviation: {self.std:3.3f}"
        )
