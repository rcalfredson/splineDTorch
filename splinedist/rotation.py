from csbdeep.utils import _raise
import cv2
import elasticdeform
import numpy as np
import random
from scipy.stats import truncnorm
import sys

import matplotlib.pyplot as plt
import random
from util import background_color, fill_in_blanks, get_border_vals, tupleMul


def split_by_channel(img):
    return tuple(img[..., i] for i in range(img.shape[-1]))


def to_int(tup):
    return tuple([int(el) for el in tup])


# returns image rotated by the given angle (in degrees, counterclockwise)
def rotate_image_with_crop(img, angle, use_linear=True):
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    print("shape of the image:", img.shape)
    print("center of the rotation:", image_center)
    mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(
        img,
        mat,
        img.shape[:2],
        flags=cv2.INTER_LINEAR if use_linear else cv2.INTER_NEAREST,
    )


def rotate_image(mat, angle, use_linear=True, crop_based_on_orig_size=False):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """
    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (
        width / 2,
        height / 2,
    )  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(
        mat,
        rotation_mat,
        (bound_w, bound_h),
        flags=cv2.INTER_LINEAR if use_linear else cv2.INTER_NEAREST,
    )
    if crop_based_on_orig_size:
        mat_center = [int(el / 2) for el in rotated_mat.shape[:2]]
        if (
            mat.shape[0] > mat.shape[1] and rotated_mat.shape[1] > rotated_mat.shape[0]
        ) or (
            mat.shape[1] > mat.shape[0] and rotated_mat.shape[0] > rotated_mat.shape[1]
        ):
            mat_center = list(reversed(mat_center))
        rotated_mat = rotated_mat[
            mat_center[0] - int(image_center[1]) : mat_center[0] + int(image_center[1]),
            mat_center[1] - int(image_center[0]) : mat_center[1] + int(image_center[0]),
        ]
        if rotated_mat.shape[0] % 2 == 1:
            rotated_mat = rotated_mat[:-1, :]
        if rotated_mat.shape[1] % 2 == 1:
            rotated_mat = rotated_mat[:, :-1]
    return rotated_mat, (bound_h, bound_w)


def sample_patches(
    data,
    patch_size,
    skip_empties=False,
    skip_partials=False,
    focused_patch_proportion=0,
    bypass=False,
):
    selected_mask = random.choice(data[0])
    if bypass:
        split_channels = split_by_channel(data[1])
        return [
            np.expand_dims(selected_mask.astype(np.uint8), axis=0),
            *[np.expand_dims(sc, axis=0) for sc in split_channels],
        ]

    len(patch_size) == selected_mask.ndim or _raise(ValueError())

    if not all((a.shape[:2] == selected_mask.shape[:2] for a in data[1:])):
        raise ValueError(
            "all input shapes must be the same: %s"
            % (" / ".join(str(a.shape) for a in data))
        )

    if not all((0 < s <= d for s, d in zip(patch_size, selected_mask.shape))):
        raise ValueError(
            "patch_size %s negative or larger than data shape %s along some dimensions"
            % (str(patch_size), str(selected_mask.shape))
        )

    # choose a random rotation angle
    rot_helper = RotationHelper(
        selected_mask.shape[0], selected_mask.shape[1], patch_size
    )
    res = None
    data = (selected_mask, data[1])
    res = rot_helper.get_random_rot_patch(
        data,
        skip_empties=skip_empties,
        skip_partials=skip_partials,
        focused_patch_proportion=focused_patch_proportion,
    )

    return res


def get_unique_vals(arr, counts_only=False):
    return np.unique(arr, return_counts=True)[-1 if counts_only else 0][1:]


def get_truncated_normal(mean=5, sd=2, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def nearest_nonzero_idx(a, x, y):
    idx = np.argwhere(a)

    # If (x,y) itself is also non-zero, we want to avoid those, so delete that
    # But, if we are sure that (x,y) won't be non-zero, skip the next step
    idx = idx[~(idx == [x, y]).all(1)]

    return idx[((idx - [x, y]) ** 2).sum(1).argmin()]


class RotationHelper:
    def __init__(self, height, width, patch_size):
        self.height = height
        self.width = width
        self.angle, self.angleRad = 0, 0
        self.patch_size = patch_size
        self.truncnorm_dist = get_truncated_normal()

    def get_rotation_angle(self):
        self.angle = np.random.randint(0, 360)
        return
        while True:
            self.angle = np.random.randint(0, 360)
            self.angleRad = np.pi * self.angle / 180
            self.aabb_h, self.aabb_w = self.calc_aabb_height_width()
            self.ht_range = self.height - self.aabb_h
            self.wd_range = self.width - self.aabb_w
            if self.ht_range > 0 and self.wd_range > 0:
                self.aabb_corner = (
                    np.random.uniform(0, self.height - self.aabb_h),
                    np.random.uniform(0, self.width - self.aabb_w),
                )
                self.aabb_center = (
                    self.aabb_corner[0] + 0.5 * self.aabb_h,
                    self.aabb_corner[1] + 0.5 * self.aabb_w,
                )
                break

    def get_focused_patch(self, arr):
        all_grays = get_unique_vals(arr)
        if len(all_grays) == 0:
            return self.get_random_patch()
        #
        # code for weighted sampling
        #
        nonzero_pixels = arr.astype(bool).astype(int)
        column_index = round(self.truncnorm_dist.rvs() * (arr.shape[1] / 10))
        row_index = round(self.truncnorm_dist.rvs() * (arr.shape[0] / 10))
        closest_gray_val_idx = nearest_nonzero_idx(
            nonzero_pixels, row_index, column_index
        )
        closest_gray_val = arr[closest_gray_val_idx[0], closest_gray_val_idx[1]]

        #
        # using equal-odds sampling
        #
        # selected_gray_val = random.choice(all_grays)

        thresh = cv2.inRange(arr, int(closest_gray_val), int(closest_gray_val))
        ret, thresh = cv2.threshold(thresh, 1, 255, 0)
        # cv2.imshow("before thresholding", cv2.resize(arr, (0, 0), fx=0.25, fy=0.25))
        # cv2.imshow("after thresholding", cv2.resize(thresh, (0, 0), fx=0.25, fy=0.25))
        # cv2.waitKey(0)
        contours, hierarchy = cv2.findContours(thresh, 1, 2)[-2:]
        if len(contours) == 0:
            return self.get_random_patch()
        cnt = contours[0]
        if len(cnt) > 20:
            M = cv2.moments(cnt)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            x, y, w, h = cv2.boundingRect(cnt)
            patch_corner = (
                max(0, round(y - 0.5 * (self.patch_size[0] - h))),
                max(0, round(x - 0.5 * (self.patch_size[1] - w))),
            )
            # input('got a non-random patch')
            return patch_corner
        else:
            return self.get_random_patch()

    def get_random_patch(self):
        y_range = self.height - self.patch_size[0]
        x_range = self.width - self.patch_size[1]
        if y_range > 0:
            rand_y = np.random.randint(0, self.height - self.patch_size[0])
        else:
            rand_y = 0
        if x_range > 0:
            rand_x = np.random.randint(0, self.width - self.patch_size[1])
        else:
            rand_x = 0
        return rand_y, rand_x

    def get_random_rot_patch(
        self, data, skip_empties=False, skip_partials=False, focused_patch_proportion=0
    ):
        deformed_mask = None
        img_to_input = cv2.cvtColor(data[1].astype(np.uint8), cv2.COLOR_RGB2BGR) / 255
        while type(deformed_mask) != np.ndarray:
            self.get_rotation_angle()
            if np.random.random_sample() <= focused_patch_proportion:
                rand_y, rand_x = self.get_focused_patch(data[0])
            else:
                # input('getting a random patch from second branch of conditional')
                rand_y, rand_x = self.get_random_patch()
            full_patch_cutoff_y = min(
                int(rand_y + self.patch_size[0]), data[1].shape[0]
            )
            full_patch_cutoff_x = min(
                int(rand_x + self.patch_size[1]), data[1].shape[1]
            )
            is_smaller_patch = full_patch_cutoff_y != int(
                rand_y + self.patch_size[0]
            ) or full_patch_cutoff_x != int(rand_x + self.patch_size[1])
            deformed_image, deformed_mask = elasticdeform.deform_random_grid(
                [img_to_input, data[0]],
                sigma=3,
                points=3,
                axis=(0, 1),
                order=[3, 0],
                crop=(
                    slice(int(rand_y), full_patch_cutoff_y),
                    slice(int(rand_x), full_patch_cutoff_x),
                ),
                rotate=self.angle,
            )
            deformed_image = np.clip(deformed_image, a_min=0, a_max=1)
            deformed_image = (255 * deformed_image).astype(np.uint8)
            bgc = background_color(deformed_image)
            if skip_empties and not np.any(deformed_mask):
                deformed_mask = None
            if (
                deformed_mask is not None
                and skip_partials
                and len(get_border_vals(deformed_mask)) > 1
            ):
                # cv2.imshow("skipped", deformed_image.astype(np.uint8))
                # print(
                # "skipping a partial mask. Values were:",
                # get_border_vals(deformed_mask),
                # )
                # cv2.waitKey(0)
                deformed_mask = None
                # cv2.imshow('good mask')
        if any([deformed_mask.shape[i] < self.patch_size[i] for i in range(2)]):
            new_img = np.full(
                (self.patch_size[0], self.patch_size[1], 3),
                fill_value=bgc,
                dtype=np.uint8,
            )
            new_img[
                self.patch_size[0] - deformed_image.shape[0] :,
                self.patch_size[1] - deformed_image.shape[1] :,
            ] = deformed_image
            new_mask = np.zeros(
                (self.patch_size[0], self.patch_size[1]), dtype=np.uint8
            )
            new_mask[
                self.patch_size[0] - deformed_image.shape[0] :,
                self.patch_size[1] - deformed_image.shape[1] :,
            ] = deformed_mask
            deformed_image = new_img
            deformed_mask = new_mask
        # print('image:', deformed_image.astype(np.uint8))
        # print('mask:', deformed_mask)
        # cv2.imshow('deformed image', cv2.resize(deformed_image.astype(np.uint8), (0, 0), fx=2.5, fy=2.5))
        # cv2.imshow('deformed mask', cv2.resize(deformed_mask, (0, 0), fx=2.5, fy=2.5))
        # cv2.waitKey(0)
        unique_vals = get_unique_vals(deformed_mask, counts_only=True)
        if len(unique_vals) == 0:
            skip_frag_checks = True
        else:
            skip_frag_checks = False
        if not skip_frag_checks:
            avg_egg_size = np.mean(unique_vals)

        sub_mask = deformed_mask
        sub_img = deformed_image
        if not skip_frag_checks:
            mask_border_vals = get_border_vals(sub_mask)
            egg_indices, egg_sizes = np.unique(sub_mask, return_counts=True)
            too_small_indices = [
                egg_indices[i]
                for i in range(len(egg_indices))
                if egg_sizes[i] <= 0.3 * avg_egg_size
                and egg_indices[i] in mask_border_vals
            ]
            too_small_mask = np.isin(sub_mask, too_small_indices)
            sub_mask[too_small_mask] = 0
        sub_img = fill_in_blanks(sub_img, bgc)
        # print('subimg:', sub_img)
        # print('submask:', sub_mask)
        # print('is smaller?', is_smaller_patch)
        # cv2.imshow('subimg', cv2.resize(sub_img.astype(np.uint8), (0, 0), fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
        # cv2.imshow('submask', cv2.resize(sub_mask, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
        # cv2.waitKey(0)
        split_channels = split_by_channel(sub_img)
        return [
            np.expand_dims(sub_mask.astype(np.uint8), axis=0),
            *[np.expand_dims(sc, axis=0) for sc in split_channels],
        ]

    def rotate_point(self, pt, center, angle):
        temp_x = pt[1] - center[1]
        temp_y = pt[0] - center[0]
        rotated_x = temp_x * np.cos(angle) - temp_y * np.sin(angle)
        rotated_y = temp_x * np.sin(angle) + temp_y * np.cos(angle)
        return (rotated_y + center[0], rotated_x + center[1])

    def calc_aabb_height_width(self):
        return (
            self.patch_size[0] * np.abs(np.cos(self.angleRad))
            + self.patch_size[1] * np.abs(np.sin(self.angleRad)),
            self.patch_size[0] * np.abs(np.sin(self.angleRad))
            + self.patch_size[1] * np.abs(np.cos(self.angleRad)),
        )
