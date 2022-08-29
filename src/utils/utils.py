import cv2
import os
import numpy as np
from utils.adjust_brightness import adjust_brightness_from_src_to_dst, read_img


import matplotlib.pyplot as plt

def center_and_resize(img, size):
    img = cv2.resize(img, (size[1], size[0]))

    return img / 127.5 - 1.0

def preprocessing(img, size):
    h, w = img.shape[:2]
    if h <= size[0]:
        h = size[0]
    else:
        x = h % 32
        h = h - x

    if w < size[1]:
        w = size[1]
    else:
        y = w % 32
        w = w - y
    # the cv2 resize func : dsize format is (W ,H)
    img = cv2.resize(img, (w, h))
    return img / 127.5 - 1.0


def load_test_data(image_path, size):
    img = cv2.imread(image_path).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = center_and_resize(img, size)
    img = np.expand_dims(img, axis=0)
    return img

def save_images(images, image_path, photo_path=None):
    fake = inverse_transform(images.squeeze())
    if photo_path:
        return imsave(adjust_brightness_from_src_to_dst(fake, read_img(photo_path)), image_path)
    else:
        return imsave(fake, image_path)


def get_image_to_save(images, photo_path):
    inv_transform_img = inverse_transform(images.squeeze())
    if photo_path:
        return adjust_brightness_from_src_to_dst(inv_transform_img, read_img(photo_path))
    else:
        return inv_transform_img


def save_image_side_by_side(real_img, generated_img, image_path, photo_path = None):
    real_img_to_save = get_image_to_save(real_img, photo_path)
    generated_img_to_save = get_image_to_save(generated_img, photo_path)

    fig, axarr = plt.subplots(1, 2, figsize=(15, 5))
    axarr[0].imshow(cv2.cvtColor(real_img_to_save, cv2.COLOR_BGR2RGB))
    axarr[1].imshow(cv2.cvtColor(generated_img_to_save, cv2.COLOR_BGR2RGB))
    plt.savefig(image_path)
    plt.close()


def save_image_side_by_side_from_path(filepath1, filepath2, output_image_path):
    img1 = cv2.imread(filepath1)
    img2 = cv2.imread(filepath2)
    cv2.imwrite(output_image_path, np.concatenate((img1, img2), axis=1))



def inverse_transform(images):
    images = (images + 1.) / 2 * 255
    # The calculation of floating-point numbers is inaccurate,
    # and the range of pixel values must be limited to the boundary,
    # otherwise, image distortion or artifacts will appear during display.
    images = np.clip(images, 0, 255)
    return images.astype(np.uint8)


def imsave(images, path):
    return cv2.imwrite(path, cv2.cvtColor(images, cv2.COLOR_BGR2RGB))



crop_image = lambda img, x0, y0, w, h: img[y0:y0 + h, x0:x0 + w]


def random_crop(img1, img2, crop_H, crop_W):
    assert img1.shape == img2.shape
    h, w = img1.shape[:2]

    # The crop width cannot exceed the original image crop width
    if crop_W > w:
        crop_W = w

    # Crop height
    if crop_H > h:
        crop_H = h

    # Randomly generate the position of the upper left corner
    x0 = np.random.randint(0, w - crop_W + 1)
    y0 = np.random.randint(0, h - crop_H + 1)

    crop_1 = crop_image(img1, x0, y0, crop_W, crop_H)
    crop_2 = crop_image(img2, x0, y0, crop_W, crop_H)
    return crop_1, crop_2

def check_create_folder(input_dir):
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    return input_dir


def str2bool(x):
    return x.lower() in ('true')
