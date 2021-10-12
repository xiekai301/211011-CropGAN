#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:42:42 2020

@author: elizabeth
"""


import numpy as np
from scipy import ndimage as ndi
import scipy
import os, sys
import SimpleITK as sitk
import argparse
import ants
from skimage.transform import resize
import glob
import re
import matplotlib.pyplot as plt

# from train_options import TrainOptions
# args = TrainOptions().parse()

IMG_HEIGHT = IMG_WIDTH = 256


def Preprocess(args):
    
    SourceDir = args.SourceDir
    first = 50 #args.first
    last = 60 #args.last
    out_dir = args.out_dir

    #location of training image volumes. Each volume needs to be in its own folder
    folders = os.listdir(SourceDir)

    #maybe you don't want to go through everything
    iterater = len(os.listdir(out_dir)) #start at 0 or where you left off
    counter = first

    #if nothing was specified for 'last' assume you want every volume in SourceDir
    if not last:
        last = len(folders)

    subset = sorted(folders)[first:last]

    #main part of processing
    for image_path in sorted(subset):
        print('working on volume: %s. Volume %d.\n' % (image_path, counter))
        counter = counter + 1

        #load image as a numpy array
        try:
            DicomImage = load_dicom_itk(os.path.join(SourceDir, image_path))
        except:
            print('Could not load Dicom information.  Skipping to next volume.\n')
            continue

        image = sitk.GetArrayFromImage(DicomImage)
        # image = image.astype(np.float32)

        #rescale the image so that each voxel is 1x1x1mm^3
        try:
            rescale_factor = DicomImage.GetSpacing()
            # image = ndi.zoom(image, rescale_factor[::-1])
            image = ndi.zoom(image, [rescale_factor[-1], IMG_HEIGHT/512, IMG_WIDTH/512])

            #I wanted to not have super long scans in this project
            if image.shape[0] > 600:
                image = image[(image.shape[0]-600):, :, :]
        except RuntimeError as err:
            print('Could not rescale image. Error message: ', err)
            continue


        #rigidly register image to template
        print('resizing...\n')
        # volume = resize(aligned_image, (256, 256, 256))
        truncted_num = image.shape[0] % 16
        image = image[:image.shape[0]-truncted_num, :, :]
        minind = image < -1024 #minimum HU is set to -1024
        maxind = image > 3000 #maximum HU is set to 3000
        image[minind] = -1024
        image[maxind] = 3000

        print('normalizing...\n')
        # image += 1024
        # image *= (1/4024) #since it is shifted to make 0 the min, normalize by 1024+3000

        path_parts = re.split("/", image_path)
        savename = path_parts[-1]
        #save the images and segmentations to .npz files.  Has to be named 'vol_data'.  Assuming no more than 9999 volumes here
        # np.savez(os.path.join(out_dir, "%04d.npz" % iterater), vol_data=volume)
        # np.savez(os.path.join(out_dir, "%s.npz" % savename), vol_data=volume)
        volout = sitk.GetImageFromArray(image.astype(np.int16))
        sitk.WriteImage(volout, "dataset/%s.nii" % savename)
        iterater = iterater+1

def load_dicom_itk(path):
    """
    read the dicom volume using SimpleITK
    """
    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()
    # reader.LoadPrivateTagsOn()  # 这一步是加载私有的元信息
    img_dicomnames = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(img_dicomnames)
    imageitk = reader.Execute()
    if imageitk.GetSpacing()[2] > 5:
        new_spacing = list(imageitk.GetSpacing())
        try:
            new_spacing[2] = float(reader.GetMetaData(2, '0018|0050'))
        except:
            new_spacing[2] = 1.0
            print("using default slice thickness of 1mm")
        imageitk.SetSpacing(new_spacing)
        
    return imageitk

if __name__ == '__main__':
    #parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--SourceDir", help="Folder with subfolders for each Dicom volume you wish to preprocess.  Remove \
                        background (outside body contour) prior to starting this process", default='data_ori')
    parser.add_argument("--first", help="Index in SourceDir to start procecessing. Default is 0", type=int, default=0)
    parser.add_argument("--last", help="Index in SourceDir to end processing. Default is the last index (all folders)", type=int, default=None)
    parser.add_argument("--out_dir", help="Folder to save your processed images", default='dataset')
    parser.add_argument("--template_dir", help="Folder containing Template Image dicom files", default='data_ori/Y170801')
    

    args = parser.parse_args()
    
    Preprocess(args)
