#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:51:09 2020

@author: elizabeth_mckenzie
"""
import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
import os
import sys
import time
import numpy as np
import glob

from crop_GAN_networks import generator as generatorNet
from crop_GAN_networks import discriminator_with_localmask as discriminatorNet
from crop_GAN_networks import load_vgg as vggNet

from crop_data import circlemask_cropped
from extraction import extract_patches, reconstruct_volume
import SimpleITK as sitk

from train_options import TrainOptions
from utils import train_step, visualize_training_output, read_nii


def train(args):
    train_data_dir = args.PATH_train
    test_data_dir = args.PATH_test
    test_img_dirs = sorted(glob.glob1(test_data_dir, "*.nii"))

    checkpoint_dir = args.checkpoint_dir
    log_dir = args.log_dir
    IMG_DEPTH = args.IMG_DEPTH
    input_shape = [args.IMG_DEPTH, args.IMG_HEIGHT, args.IMG_WIDTH, 1]
    IMG_STRIDE = args.IMG_STRIDE
    EPOCHS = args.EPOCHS
    GPU = args.GPU
    continue_training = args.continue_training
    checkpoint_restore_dir = args.checkpoint_restore_dir
    step = 0  # for tensorboard

    ###define optimizers###
    generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
    discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)

    ###prepare gpu usage###
    gpu = '/device:GPU:%d' % GPU
    with tf.device(gpu):
        ###load networks###
        generator = generatorNet()
        discriminator = discriminatorNet()
        Vgg = vggNet()

        ###Setup Checkpoints##
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                         discriminator_optimizer=discriminator_optimizer,
                                         generator=generator,
                                         discriminator=discriminator)

        epoch_restart = 0
        if continue_training:
            status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_restore_dir))
            epoch_restart = int(status._checkpoint.save_path_string.split('-')[-1])
        ###Training loop###
        current_time = '{}{:0>2d}{:0>2d}'.format(*time.localtime()[:3])  # use date to logs
        summary_writer = tf.summary.create_file_writer(log_dir + current_time + '/training')
        summary_val_writer = tf.summary.create_file_writer(log_dir + current_time + '/validation')

        # for epoch in range(EPOCHS):
        for epoch in range(epoch_restart, EPOCHS):
            start = time.time()
            print("Epoch: ", epoch)
            # for each epoch iterate over subset of dataset
            # for n, img_dir in enumerate(sorted(os.listdir(train_data_dir))):
            for n, nii_name in enumerate(sorted(glob.glob1(train_data_dir, "*.nii"))):
                # if n < 1:
                #     continue
                print("\n--------img_dir: ", nii_name)
                nii_path = os.path.join(train_data_dir, nii_name)
                img = read_nii(nii_path)
                nii_edge_path = nii_path.replace('.nii', '_edge.nii.gz')
                edge = read_nii(nii_edge_path)
                patches = extract_patches(img=img, patch_shape=input_shape[:3], extraction_step=[IMG_STRIDE, 0, 0])
                edge_patches = extract_patches(img=edge, patch_shape=input_shape[:3], extraction_step=[IMG_STRIDE, 0, 0])
                for l in range(patches.shape[0]):
                # for l in range(1):
                    patch = patches[l, :]  # patch排序
                    edge_patch = edge_patches[l, :]
                    mask = circlemask_cropped(input_shape)
                    patch = np.expand_dims(patch, axis=[0,-1])  # 加第一个轴
                    edge_patch = np.expand_dims(edge_patch, axis=[0,-1])  # 加第一个轴
                    target = patch
                    input_image = target.copy()
                    input_image[mask] = -1
                    target = tf.convert_to_tensor(target, dtype=tf.float16, name='target')
                    mask = tf.convert_to_tensor(mask, dtype=tf.float16, name='mask')
                    input_image = tf.convert_to_tensor(input_image, dtype=tf.float16, name='input_image')
                    # training step
                    loss, pred_img, gen_vgg, tar_vgg = train_step(input_image, target, mask, epoch, generator, discriminator, Vgg,
                                               generator_optimizer, discriminator_optimizer, n, l, training=True)
                if (n + 1) % 20 == 0:
                    summary_images, summary_images_act = visualize_training_output(input_image, target, pred_img,
                                                                                gen_vgg, tar_vgg, IMG_DEPTH)
                    # print()
                    step = step + 1
                    # log to Tensorboard
                    with summary_writer.as_default():
                        tf.summary.image("Training visualization", summary_images, step=step)
                        tf.summary.image("Training Activation visualization", summary_images_act, step=step)
                        tf.summary.scalar('gen_total_loss', loss[0], step=step)
                        tf.summary.scalar('gen_gan_loss', loss[1], step=step)
                        tf.summary.scalar('gen_l1_loss', loss[2], step=step)
                        tf.summary.scalar('vgg_loss', loss[3], step=step)
                        tf.summary.scalar('fm_loss', loss[4], step=step)
                        tf.summary.scalar('disc_loss', loss[5], step=step)

                    # log val data to Tensorboard
                    num1 = np.random.randint(0, len(test_img_dirs))
                    nii_val_path = os.path.join(test_data_dir, test_img_dirs[num1])
                    img_val = read_nii(nii_val_path)
                    patches_val = extract_patches(img=img_val, patch_shape=input_shape[:3], extraction_step=[IMG_STRIDE, 0, 0])
                    # num2 slice from one random patient
                    num2 = np.random.randint(0, len(patches_val))
                    patch_val = patches_val[num2, :]  # patch排序
                    mask_val = circlemask_cropped(input_shape)
                    patch_val = np.expand_dims(patch_val, axis=[0,-1])  # 加第一个轴
                    target_val = patch_val
                    input_image_val = target_val.copy()
                    input_image_val[mask_val] = -1
                    target_val = tf.convert_to_tensor(target_val, dtype=tf.float16, name='target_val')
                    mask_val = tf.convert_to_tensor(mask_val, dtype=tf.float16, name='mask_val')
                    input_image_val = tf.convert_to_tensor(input_image_val, dtype=tf.float16, name='input_image_val')
                    # create val loss info
                    loss_val, pred_img, gen_vgg, tar_vgg = train_step(input_image_val, target_val, mask_val, epoch,
                                                                      generator, discriminator, Vgg, generator_optimizer,
                                                                      discriminator_optimizer, num1, num2, training=False)

                    # create summary image of results
                    summary_images, summary_images_act = visualize_training_output(input_image_val, target_val, pred_img,
                                                                                   gen_vgg, tar_vgg, IMG_DEPTH)

                    with summary_val_writer.as_default():
                        tf.summary.image("Validation visualization", summary_images, step=step)
                        tf.summary.image("Validation Activation visualization", summary_images_act, step=step)
                        tf.summary.scalar('gen_total_loss', loss_val[0], step=step)
                        tf.summary.scalar('gen_gan_loss', loss_val[1], step=step)
                        tf.summary.scalar('gen_l1_loss', loss_val[2], step=step)
                        tf.summary.scalar('vgg_loss', loss_val[3], step=step)
                        tf.summary.scalar('fm_loss', loss_val[4], step=step)
                        tf.summary.scalar('disc_loss', loss_val[5], step=step)

            # save checkpoint every 20 epochs
            if (epoch + 1) % 1 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

            print('Time taken for epoch {} is {:.2f} sec\n'.format(epoch + 1, time.time() - start))
        checkpoint.save(file_prefix=checkpoint_prefix)


def test(args):

    cropped_savepath = args.cropped_savepath
    prediction_savepath = args.prediction_savepath
    uncropped_savepath = args.uncropped_savepath
    mask_savepath = args.mask_savepath
    save_paths = [prediction_savepath, cropped_savepath, uncropped_savepath, mask_savepath]

    GPU = args.GPU
    checkpoint_restore_dir = args.checkpoint_restore_dir

    test_data_dir = args.PATH_test
    input_shape = [args.IMG_DEPTH, args.IMG_HEIGHT, args.IMG_WIDTH, 1]
    IMG_STRIDE = args.IMG_STRIDE

    # load testing dataset
    print('loading test dataset')


    gpu = '/gpu:%d' % GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
    with tf.device(gpu):

        ###load trained networks###
        print('loading saved networks')
        generator = generatorNet()
        checkpoint = tf.train.Checkpoint(generator=generator)
        status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_restore_dir))

        # for n, img_dir in enumerate(sorted(os.listdir(test_data_dir))):
        for n, img_dir in enumerate(sorted(glob.glob1(test_data_dir, '*.nii'))):
            # if n > 0:
            #     continue
            print("img_dir: ", img_dir)
            nii_path = os.path.join(test_data_dir, img_dir)
            img = sitk.ReadImage(nii_path)
            img = sitk.GetArrayFromImage(img)
            truncted_num = img.shape[0] % 16
            img = img[:img.shape[0] - truncted_num, :, :]
            img[img < 0] = 0
            img[img > 3000] = 3000
            img = 2 * img / 3000 - 1  # # nii follewed by matlab
            patches = extract_patches(img=img, patch_shape=input_shape[:3], extraction_step=[IMG_STRIDE, 0, 0])
            start = time.time()
            for l in range(patches.shape[0]):
                patch = patches[l, :]  # patch排序
                mask = circlemask_cropped(input_shape)
                mask = np.expand_dims(mask, axis=0)
                patch = np.expand_dims(patch, axis=[0,-1])  # 加第一个轴
                target = patch
                input_image = target.copy()
                input_image[mask] = -1
                mask = tf.convert_to_tensor(mask, dtype=tf.float16, name='mask')
                input_image = tf.convert_to_tensor(input_image, dtype=tf.float16, name='input_image')

                prediction = generator(input_image, training=False)

                prediction = (prediction * mask) + (input_image * (1 - mask))
                patches[l, :] = prediction.numpy()[0, ..., 0]  # synthetically uncropped

            recon_img = reconstruct_volume(patches=patches, expected_shape=img.shape, extraction_step=(16, 1, 1))
            recon_img = (recon_img * 0.5) + 0.5
            # recon_img = (recon_img * 4024) - 1024
            recon_img = recon_img * 3000
            volout = sitk.GetImageFromArray(recon_img.astype(np.int16))
            sitk.WriteImage(volout, "dataset/%s_testout.nii" % img_dir[:-4])
            print('Time taken for prediction {} is {:.2f} sec\n'.format(n, time.time() - start))


if __name__ == '__main__':
    args = TrainOptions().parse()
    if args.continue_training and args.checkpoint_restore_dir is None:
        print('need to provide checkpoint restore directory (checkpoint_restore_dir) if continuing training')
        sys.exit()

    if args.Training:
        if not os.path.isdir(args.checkpoint_dir):
            os.mkdir(args.checkpoint_dir)
        if not os.path.isdir(args.log_dir):
            os.mkdir(args.log_dir)
        train(args)
    else:
        if not os.path.isdir(args.cropped_savepath):
            os.mkdir(args.cropped_savepath)
        if not os.path.isdir(args.prediction_savepath):
            os.mkdir(args.prediction_savepath)
        if not os.path.isdir(args.uncropped_savepath):
            os.mkdir(args.uncropped_savepath)
        if not os.path.isdir(args.mask_savepath):
            os.mkdir(args.mask_savepath)
        test(args)
