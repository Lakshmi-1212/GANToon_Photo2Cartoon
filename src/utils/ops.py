import tensorflow as tf
import numpy as np
from utils.vgg19 import Vgg19



def relu(x):
    return tf.nn.relu(x)


##################################################################################
# Loss function
##################################################################################

def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))
    return loss


def Huber_loss(x, y):
    h = tf.keras.losses.Huber()
    return h(x, y)


def calculate_mse(value):
    return tf.reduce_mean(tf.square(value))


def compute_disc_adv_loss(loss_func, real, gray, fake, real_blur, step, writer, real_loss_weight,
                           fake_loss_weight, gray_loss_weight, real_blur_loss_weight):
    real_loss = 0
    gray_loss = 0
    fake_loss = 0
    real_blur_loss = 0

    if loss_func == 'wgan-gp' or loss_func == 'wgan-lp':
        real_loss = -tf.reduce_mean(real)
        gray_loss = tf.reduce_mean(gray)
        fake_loss = tf.reduce_mean(fake)
        real_blur_loss = tf.reduce_mean(real_blur)

    if loss_func == 'lsgan':
        real_loss = calculate_mse(real - 1.0)
        gray_loss = calculate_mse(gray)
        fake_loss = calculate_mse(fake)
        real_blur_loss = calculate_mse(real_blur)

    if loss_func == 'gan' or loss_func == 'dragan':
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
        gray_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(gray), logits=gray))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))
        real_blur_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(real_blur), logits=real_blur))

    if loss_func == 'hinge':
        real_loss = tf.reduce_mean(relu(1.0 - real))
        gray_loss = tf.reduce_mean(relu(1.0 + gray))
        fake_loss = tf.reduce_mean(relu(1.0 + fake))
        real_blur_loss = tf.reduce_mean(relu(1.0 + real_blur))

    loss = real_loss_weight * real_loss + fake_loss_weight * fake_loss + \
           gray_loss_weight * gray_loss + real_blur_loss_weight * real_blur_loss

    with writer.as_default(step=step):
        """" Summary """
        tf.summary.scalar("Discriminator_adv_real_loss", real_loss)
        tf.summary.scalar("Discriminator_adv_fake_loss", fake_loss)
        tf.summary.scalar("Discriminator_adv_gray_loss", gray_loss)
        tf.summary.scalar("Discriminator_adv_real_blur_loss", real_blur_loss)

    return loss


def compute_disc_t_loss(real, fake):

    # 'lsgan' - least square mean loss
    real_loss = calculate_mse(real - 1.0)
    fake_loss = calculate_mse(fake)

    d_loss = 0.5 * (real_loss + fake_loss)
    return d_loss


def compute_generator_adv_loss(loss_func, fake):
    fake_loss = 0

    if loss_func == 'wgan-gp' or loss_func == 'wgan-lp':
        fake_loss = -tf.reduce_mean(fake)

    if loss_func == 'lsgan':
        fake_loss = calculate_mse(fake - 1.0)

    if loss_func == 'gan' or loss_func == 'dragan':
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

    if loss_func == 'hinge':
        fake_loss = -tf.reduce_mean(fake)

    loss = fake_loss

    return loss


def get_feature_map(pre_trained_model, model_type, input):

    return pre_trained_model(input, training=False)

def compute_content_loss(pre_train_model, model_type, real, fake):
    real_feature_map = get_feature_map(pre_train_model, model_type, real)
    fake_feature_map = get_feature_map(pre_train_model, model_type, fake)

    loss = L1_loss(real_feature_map, fake_feature_map)

    return loss


def random_color_shift(images, mode='uniform'):
    if mode == 'normal':
        b_weight = tf.random.normal(shape=[1], mean=0.114, stddev=0.1)
        g_weight = tf.random.normal(shape=[1], mean=0.587, stddev=0.1)
        r_weight = tf.random.normal(shape=[1], mean=0.299, stddev=0.1)
    elif mode == 'uniform':
        b_weight = tf.random.uniform(shape=[1], minval=0.014, maxval=0.214)
        g_weight = tf.random.uniform(shape=[1], minval=0.487, maxval=0.687)
        r_weight = tf.random.uniform(shape=[1], minval=0.199, maxval=0.399)

    output_rcs = []
    for image in images:
        r,g,b = tf.split(image, num_or_size_splits=3, axis=3)
        output = (b_weight * b + g_weight * g + r_weight * r) / (b_weight + g_weight + r_weight)
        output_rcs.append(output)

    return output_rcs


def textural_representation(images, mode='uniform'):
    return random_color_shift(images, mode)


def compute_generator_texture_loss(fake_t_rep):
    # Use 'lsgan' least squares loss function
    texture_fake_loss = calculate_mse(fake_t_rep - 1.0)
    return texture_fake_loss


def feature_map_model_init(feature_map_model_type='VGG19'):
    inputs = tf.keras.Input([256, 256, 3])

    p_model = None
    if (feature_map_model_type == 'MOBILENETV2'):
        print(f'4DEBUG: Initializing with MOBILENETV2 model')
        model = tf.keras.applications.MobileNetV2(
            include_top=False,
            alpha=1.3,
            weights='imagenet',
            input_tensor=inputs,
            pooling=None,
            classes=1000)
        p_model: tf.keras.Model = tf.keras.Model(
            inputs,
            model.get_layer('block_6_expand').output)
    elif (feature_map_model_type == 'VGG19') :
        model = tf.keras.applications.vgg19.VGG19(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs,
            pooling=None,
            classes=1000
        )

        print(f'DEBUG: Initializing with VGG19 model')
        p_model: tf.keras.Model = tf.keras.Model(
            inputs,
            model.get_layer('block4_conv4').output)

    else:
        raise(f'Feature map model initialization failed!! Unhandled model type: {feature_map_model_type}')
    p_model.trainable = False
    return p_model


def compute_color_loss_anime(con, fake):
    con = rgb2yuv(con)
    fake = rgb2yuv(fake)

    return L1_loss(con[:, :, :, 0], fake[:, :, :, 0]) + Huber_loss(con[:, :, :, 1], fake[:, :, :, 1]) + Huber_loss(
        con[:, :, :, 2], fake[:, :, :, 2])


def convert_rgb2hsv(rgb_img):
    rgb_img = (rgb_img + 1.0) / 2.0
    hsv_img = tf.image.rgb_to_hsv(rgb_img)
    return hsv_img


def get_hue_from_hsv(hsv_img):
    return hsv_img[:, :, 0]


# Color loss using the hue component of the image
def compute_color_loss(real_rgb, fake_rgb):
    real_hue = get_hue_from_hsv(convert_rgb2hsv(real_rgb))
    fake_hue = get_hue_from_hsv(convert_rgb2hsv(fake_rgb))

    min_hue = tf.minimum(fake_hue - real_hue, fake_hue + real_hue)
    col_loss = calculate_mse(min_hue - 1.0)

    return col_loss


def compute_total_variation_loss(inputs):
    """
    A smooth loss in fact. Like the smooth prior in MRF.
    V(y) = || y_{n+1} - y_n ||_2
    """
    dh = inputs[:, :-1, ...] - inputs[:, 1:, ...]
    dw = inputs[:, :, :-1, ...] - inputs[:, :, 1:, ...]
    size_dh = tf.size(dh, out_type=tf.float32)
    size_dw = tf.size(dw, out_type=tf.float32)
    return tf.nn.l2_loss(dh) / size_dh + tf.nn.l2_loss(dw) / size_dw


def rgb2yuv(rgb):
    """
    Convert RGB image into YUV
    """
    rgb = (rgb + 1.0) / 2.0
    return tf.image.rgb_to_yuv(rgb)
