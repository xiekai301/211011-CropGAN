#coding=utf-8
#-------------------------------------------------------------------------------
# Name:        extration_3D_patches reconstruction_3D
# Purpose:
#
# Author:      SunHF
#
# Created:     04/07/2021
# Copyright:   (c) SunHF 2021
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
import numbers
from numpy.lib.stride_tricks import as_strided
import itertools


def extract_patches(img, patch_shape, extraction_step):            # def extract_patches(array, 64, 32):
    arr_ndim = img.ndim
    patch_strides = img.strides
    if isinstance(patch_shape, numbers.Number):   #检验变量是否为数字
        patch_shape = tuple([patch_shape] * arr_ndim)
    if isinstance(extraction_step, numbers.Number):
        extraction_step = tuple([extraction_step] * arr_ndim)
    slices = tuple(slice(None, None, st) for st in [extraction_step[0]])
    indexing_strides = img[slices].strides
    patch_indices_shape = (
        (np.array(img.shape) - np.array(patch_shape)) // np.array(extraction_step)
    ) + 1
    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))
    patches = as_strided(img, shape=shape, strides=strides)
    ndim = len(img.shape)
    npatches = np.prod(patches.shape[:ndim])
    patches_ = patches.reshape([npatches, ] + patch_shape)
    return patches_


def generate_indexes(patch_shape, expected_shape, pad_shape):

    ndims = len(patch_shape)
    # idxs = [range(0, expected_shape[i] - pad_shape[i], pad_shape[i]) for i in range(ndims-1)]
    idxs = [range(0, expected_shape[i] - pad_shape[i], pad_shape[i]) for i in range(ndims-3)]
    return itertools.product(*idxs)


# def reconstruct_volume(patches, expected_shape, extraction_step=(32, 32, 32)):
#     patch_shape = patches.shape
#     assert len(patch_shape) - 1 == len(expected_shape)
#     reconstructed_img = np.zeros(expected_shape, dtype='float32')
#     for count, coord in enumerate(generate_indexes(patch_shape, expected_shape, extraction_step)) :
#         selection = [slice(coord[i], coord[i] + patch_shape[i+1]) for i in range(len(coord))]
# #        print('count: ', count)
# #        print('coord: ', coord)
# #        print ('selection: ', selection)
#         reconstructed_img[selection] = patches[count, :]
#     return reconstructed_img

def reconstruct_volume(patches, expected_shape, extraction_step=(32, 32, 32)):

    patch_shape = patches.shape
    assert len(patch_shape) - 1 == len(expected_shape)
    reconstructed_img = np.zeros(expected_shape, dtype='float32')
    reconstructed_index = np.zeros(expected_shape, dtype='float32')

    for count, coord in enumerate(generate_indexes(patch_shape, expected_shape, extraction_step)) :
        selection = [slice(coord[i], coord[i] + patch_shape[i+1]) for i in range(len(coord))]
        reconstructed_img[selection] += patches[count, :]
        reconstructed_index[selection] += 1

    return reconstructed_img/reconstructed_index




