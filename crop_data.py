#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:23:35 2020

@author: elizabeth_mckenzie
"""

import numpy as np
from train_options import TrainOptions
import matplotlib.pyplot as plt

args = TrainOptions().parse()
order = 0
# input_shape = (128, 128, 128, 1)
input_shape = (args.IMG_DEPTH, args.IMG_HEIGHT, args.IMG_WIDTH, 1)


def circlemask_cropped(input_shape):
    # D, H, W, _ = x.shape
    D, H, W, _ = input_shape
    x, y = np.ogrid[:H, :W]
    cx, cy = H / 2, W / 2
    radius = int(np.random.uniform(0.75, 0.75) * H / 2)
    r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
    circmask = r2 > radius * radius
    mask = np.expand_dims(circmask, axis=[0,1,-1]).repeat([D, ], axis=1)
    return mask

