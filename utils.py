import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

from crop_gan_losses import generator_loss_with_features
from crop_gan_losses import discriminator_wasshinge_loss as discriminator_loss

import numpy as np
from matplotlib import pyplot as plt
import io
import SimpleITK as sitk

def read_nii(nii_path):
    img = sitk.ReadImage(nii_path)
    img = sitk.GetArrayFromImage(img)
    truncted_num = img.shape[0] % 16
    img = img[:img.shape[0] - truncted_num, :, :]
    img[img < 0] = 0
    img[img > 3000] = 3000
    img = 2 * img / 3000 - 1  # # nii follewed by matlab
    return img

def train_step(input_image, target, mask, epoch, generator, discriminator, vgg_net, generator_optimizer,
               discriminator_optimizer, n, l, training=True):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # output from generator
        gen_output = generator(input_image, training=training)

        # replace known region with input
        # gen_output = (gen_output * mask) + (input_image * (1 - mask))  # mask is 1 in background, 0 in known region
        #
        # outputs from VGG model
        gen_vgg = vgg_net(gen_output)
        tar_vgg = vgg_net(target)

        # discriminator on real target image
        disc_real_output, target_disc_layers = discriminator([input_image, target, mask], training=training)
        print('disc real output: ', tf.math.reduce_mean(disc_real_output).numpy())

        # discriminator on fake target image
        disc_generated_output, gen_disc_layers = discriminator([input_image, gen_output, mask], training=training)
        print('disc gen output: ', tf.math.reduce_mean(disc_generated_output).numpy())

        # the losses from the generator
        gen_total_loss, gen_gan_loss, gen_l1_loss, vgg_loss, fm_loss = generator_loss_with_features(
            disc_generated_output, gen_output, target, tar_vgg, gen_vgg, target_disc_layers, gen_disc_layers)
        # the losses from the discriminator
        disc_loss, loss_on_generated, loss_on_real = discriminator_loss(disc_real_output, disc_generated_output)

    if training:
        # calculate the gradients of the loss
        generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)

        # calculate the gradients of the loss
        discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        # apply these gradients to the optimizers
        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

        # apply these gradients to the optimizers
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

        print(
            'Epoch {} Patient {} Patch {} -- G_total: {:.3f}\tG_gan: {:.3f}\tG_l1: {:.3f}\tG_vgg: {:.3f}\t'
            'G_fm: {:.3f}\tD_total: {:.3f}\tD_real: {:.3f}\tD_fake: {:.3f}'.format(
                epoch, n, l, gen_total_loss, gen_gan_loss, gen_l1_loss, vgg_loss, fm_loss, disc_loss, loss_on_real,
                loss_on_generated))

    return (gen_total_loss, gen_gan_loss, gen_l1_loss, vgg_loss, fm_loss, disc_loss), gen_output, gen_vgg, tar_vgg


# def visualize_training_output(input_image, target, pred_img, gen_vgg, tar_vgg):
def visualize_training_output(input_image, target, pred_img, gen_vgg, tar_vgg, IMG_DEPTH):
    # visualize results on testing dataset
    # example_input = (input_image[0, :, :, 64, 0].numpy() + 1.0) / 2.0  # cropped
    # example_target = (target[0, :, :, 64, 0].numpy() + 1.0) / 2.0  # uncropped
    # pred_img = tf.clip_by_value(pred_img, -1.0, 1.0)
    # example_pred = (pred_img[0, :, :, 64, 0].numpy() + 1.0) / 2.0  # predicted completed image
    example_input = (input_image[0, IMG_DEPTH // 2, :, :, 0].numpy() + 1.0) / 2.0  # cropped
    example_target = (target[0, IMG_DEPTH // 2, :, :, 0].numpy() + 1.0) / 2.0  # uncropped
    pred_img = tf.clip_by_value(pred_img, -1.0, 1.0)
    example_pred = (pred_img[0, IMG_DEPTH // 2, :, :, 0].numpy() + 1.0) / 2.0  # predicted completed image

    figure = image_grid(example_input, example_target, example_pred)
    summary_image = plot_to_image(figure)
    figure_act = layers_image_grid(gen_vgg, tar_vgg)
    summary_image_act = plot_to_image(figure_act)
    return summary_image, summary_image_act


def image_grid(test_input, tar, pred):
    figure = plt.figure(figsize=(15, 6))

    display_list = [test_input, tar, pred]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # print(display_list[i])
        # plt.imshow(np.flip((display_list[i] * 255).astype('uint8')), cmap='gray', vmin=0, vmax=255)
        plt.imshow((display_list[i] * 255).astype('uint8'), cmap='gray', vmin=0, vmax=255)
        plt.axis('off')

    return figure


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


def layers_image_grid(gen_vgg, tar_vgg):
    figure = plt.figure(figsize=(15, 6))
    cols = len(tar_vgg)
    g_title = ['Predicted: layer %d' % i for i in range(len(gen_vgg))]
    t_title = ['Target: layer %d' % i for i in range(len(tar_vgg))]

    for i, g in enumerate(gen_vgg):
        channel_image = process_activations(g)
        plt.subplot(2, cols, i + 1)
        plt.title(g_title[i])
        # plt.imshow(np.flip(channel_image), cmap='viridis')
        plt.imshow(channel_image, cmap='viridis')
        plt.axis('off')

    for j, t in enumerate(tar_vgg):
        channel_image = process_activations(t)
        plt.subplot(2, cols, j + cols + 1)
        plt.title(t_title[j])
        # plt.imshow(np.flip(channel_image), cmap='viridis')
        plt.imshow(channel_image, cmap='viridis')
        plt.axis('off')

    return figure


def process_activations(layer_activation):
    avg_activation = np.nanmean(layer_activation, -1)
    size = layer_activation.shape[1]
    depth = layer_activation.shape[3]
    # channel_image = avg_activation[0, :, :, depth // 2]
    channel_image = avg_activation[0, size // 2, :, :]
    channel_image -= channel_image.mean()
    channel_image /= channel_image.std()
    channel_image *= 64
    channel_image += 128
    channel_image = np.clip(channel_image, 0, 255).astype('uint8')

    return channel_image
