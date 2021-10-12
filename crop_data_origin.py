#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:23:35 2020

@author: elizabeth_mckenzie
"""

import numpy as np
from scipy.ndimage.interpolation import rotate as ndrotate
import tensorflow as tf
from skimage.transform import resize
import matplotlib.pyplot as plt

order = 0
# input_shape = (128, 128, 128, 1)
input_shape = (32, 256, 256, 1)


def first_rot(ubermask, angle_xy):
    ubermask = ndrotate(ubermask.numpy(), angle_xy.numpy(), axes=(0, 1), reshape=False, order=order)
    return ubermask


def second_rot(ubermask, angle_xz):
    ndrotate(ubermask.numpy(), angle_xz.numpy(), axes=(0, 2), reshape=False, order=order)
    return ubermask


def third_rot(ubermask, angle_yz):
    ndrotate(ubermask.numpy(), angle_yz.numpy(), axes=(1, 2), reshape=False, order=order)
    return ubermask


def crop_the_mask(crop_mask, crop_amount, top):
    # generate mask for cropping and crop to zeros
    crop_mask = crop_mask.numpy()
    if top:
        crop_top = crop_amount
        crop_mask[0:crop_top, :, :] = 0.0
    else:
        crop_bottom = crop_amount
        crop_mask[-crop_bottom:, :, :] = 0.0
    return crop_mask


def rotate_and_crop(x, crop_amount, top=True):
    x = tf.cast(x, dtype=tf.float64)
    # generate random angles for rotation.  Most is in chin up/down direction
    # angle_xy = tf.random.uniform([], minval=-45, maxval=0, dtype=tf.dtypes.int32)
    # angle_xz = tf.random.uniform([], minval=-5, maxval=5, dtype=tf.dtypes.int32)
    # angle_yz = tf.random.uniform([], minval=-5, maxval=5, dtype=tf.dtypes.int32)
    angle_xy = tf.constant(0, dtype=tf.dtypes.int32)
    angle_xz = tf.constant(0, dtype=tf.dtypes.int32)
    angle_yz = tf.constant(0, dtype=tf.dtypes.int32)

    ubermask = tf.ones_like(x, dtype=tf.float64)  # mask for the mask

    # rotate mask's mask
    ubermask = tf.py_function(func=first_rot, inp=[ubermask, angle_xy], Tout=tf.float64)
    ubermask = tf.py_function(func=second_rot, inp=[ubermask, angle_xz], Tout=tf.float64)
    ubermask = tf.py_function(func=third_rot, inp=[ubermask, angle_yz], Tout=tf.float64)

    # generate mask for cropping and crop to zeros
    crop_mask = tf.ones_like(ubermask, dtype=tf.float64)

    crop_mask = tf.py_function(func=crop_the_mask, inp=[crop_mask, crop_amount, top], Tout=tf.float64)

    # crop image
    cropped_mask = ubermask * crop_mask

    # rotate back
    cropped_mask = tf.py_function(func=third_rot, inp=[cropped_mask, -angle_yz], Tout=tf.float64)
    cropped_mask = tf.py_function(func=second_rot, inp=[cropped_mask, -angle_xz], Tout=tf.float64)
    cropped_mask = tf.py_function(func=first_rot, inp=[cropped_mask, -angle_xy], Tout=tf.float64)

    output_img = x * cropped_mask

    output_img = tf.cast(output_img, dtype=tf.float16)
    cropped_mask = tf.cast(cropped_mask, dtype=tf.float16)

    return (output_img, cropped_mask)


def circlemask_cropped(x):
    D, H, W, _ = x.shape
    x, y = np.ogrid[:H, :W]
    cx, cy = H / 2, W / 2
    radius = int(np.random.uniform(0.7, 0.7) * H / 2)
    r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
    circmask = r2 <= radius * radius
    mask = np.expand_dims(circmask, axis=[0,-1]).repeat([D, ], axis=0)
    return mask

# def crop_data(x):
#     x.set_shape(input_shape)  # hardcoded in here to get .map(tffunction) to work
#
#     # generate cropping amounts
#     crop_bottom = 50  # tf.random.uniform([], minval=40, maxval=70, dtype=tf.dtypes.int32)
#     top_max = (x.shape[1] - crop_bottom) - 35  # (128 -70) = 58, - 35 = 23 as max
#     # crop_top = tf.random.uniform([], minval=10, maxval=top_max,
#     #                              dtype=tf.dtypes.int32)  # want to guarentee there is something left
#     crop_top = tf.constant(30, dtype=tf.dtypes.int32)
#
#     (cropped_top_image, mask_Top) = rotate_and_crop(x, crop_top, top=True)
#     (cropped_bottom_image, mask_Bottom) = rotate_and_crop(cropped_top_image, crop_bottom, top=False)
#
#     final_cropped_image = cropped_bottom_image
#     final_cropped_mask = mask_Top * mask_Bottom
#     return (final_cropped_image, x, final_cropped_mask)



def crop_data(x):
    x = tf.cast(x, dtype=tf.float16)
    x.set_shape(input_shape)  # hardcoded in here to get .map(tffunction) to work
    cropped_mask = tf.py_function(func=circlemask_cropped, inp=[x], Tout=tf.float16)
    cropped_image = x * cropped_mask
    cropped_image = tf.cast(cropped_image, dtype=tf.float16)
    cropped_mask = tf.cast(cropped_mask, dtype=tf.float16)
    return (cropped_image, x, cropped_mask)


npy_name = 'dataset/train/Y170109.npz'
img_files = np.load(npy_name)
img = img_files['vol_data']
img = resize(img, (32,256,256))
# img = np.expand_dims(img, axis=-1)
# x = tf.convert_to_tensor(img)
# y = crop_data(x)

# plt.imshow(final_cropped_mask.numpy()[64,:,:].astype(np.float64), cmap='gray'); plt.show()

