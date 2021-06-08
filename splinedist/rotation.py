from csbdeep.utils import _raise
import cv2
import elasticdeform
import numpy as np

import matplotlib.pyplot as plt


def split_by_channel(img):
    return tuple(img[..., i] for i in range(img.shape[-1]))


def to_int(tup):
    return tuple([int(el) for el in tup])


def rotate_image(mat, angle, use_linear=True):
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
    return rotated_mat, (bound_h, bound_w)


def sample_patches(data, patch_size, skip_empties=False):
    len(patch_size) == data[0].ndim or _raise(ValueError())

    if not all((a.shape[:2] == data[0].shape[:2] for a in data)):
        raise ValueError(
            "all input shapes must be the same: %s"
            % (" / ".join(str(a.shape) for a in data))
        )

    if not all((0 < s <= d for s, d in zip(patch_size, data[0].shape))):
        raise ValueError(
            "patch_size %s negative or larger than data shape %s along some dimensions"
            % (str(patch_size), str(data[0].shape))
        )

    # choose a random rotation angle
    rot_helper = RotationHelper(data[0].shape[0], data[0].shape[1], patch_size)
    res = None
    res = rot_helper.get_random_rot_patch(data, skip_empties=skip_empties)
        
    return res


class RotationHelper:
    def __init__(self, height, width, patch_size):
        self.height = height
        self.width = width
        self.angle, self.angleRad = 0, 0
        self.patch_size = patch_size

    def get_rotation_angle(self):
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

    def get_random_rot_patch(self, data, skip_empties=False):
        deformed_mask = None
        while type(deformed_mask) != np.ndarray:
            self.get_rotation_angle()
            rand_y = np.random.randint(0, self.height - self.patch_size[0])
            rand_x = np.random.randint(0, self.width - self.patch_size[1])
            deformed_image, deformed_mask = elasticdeform.deform_random_grid(
                [data[1], data[0]],
                sigma=6,
                points=3,
                axis=(0, 1),
                order=[3, 0],
                crop=(
                    slice(int(rand_y), int(rand_y + self.patch_size[0])),
                    slice(int(rand_x), int(rand_x + self.patch_size[1])),
                ),
                rotate=self.angle,
            )
            # how to check for empty patches? Could check if every element of the mask
            # equals zero?
            if skip_empties and not np.any(deformed_mask):
                deformed_mask = None
        # print('image:', deformed_image.astype(np.uint8))
        # print('mask:', deformed_mask)
        # cv2.imshow('deformed image', cv2.resize(deformed_image.astype(np.uint8), (0, 0), fx=2.5, fy=2.5))
        # cv2.imshow('deformed mask', cv2.resize(deformed_mask, (0, 0), fx=2.5, fy=2.5))
        # cv2.waitKey(0)
        unique_vals = np.unique(deformed_mask, return_counts=True)[-1][1:]
        if len(unique_vals) == 0:
            skip_frag_checks = True
        else:
            skip_frag_checks = False
        if not skip_frag_checks:
            avg_egg_size = np.mean(np.unique(deformed_mask, return_counts=True)[-1][1:])

        sub_mask = deformed_mask
        sub_img = deformed_image
        if not skip_frag_checks:
            mask_border_vals = set(
                np.concatenate(
                    (
                        sub_mask[[0, sub_mask.shape[0] - 1], :].ravel(),
                        sub_mask[1:-1, [0, sub_mask.shape[1] - 1]].ravel(),
                    )
                )
            )
            egg_indices, egg_sizes = np.unique(sub_mask, return_counts=True)
            too_small_indices = [
                egg_indices[i]
                for i in range(len(egg_indices))
                if egg_sizes[i] <= 0.3 * avg_egg_size
                and egg_indices[i] in mask_border_vals
            ]
            too_small_mask = np.isin(sub_mask, too_small_indices)
            sub_mask[too_small_mask] = 0
        print('subimg:', sub_img)
        print('submask:', sub_mask)
        cv2.imshow('subimg', sub_img.astype(np.uint8))
        cv2.imshow('submask', cv2.resize(sub_mask, (0, 0), fx=3, fy=3))
        cv2.waitKey(0)
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
