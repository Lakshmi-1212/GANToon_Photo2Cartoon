import tensorflow as tf
import argparse
from utils.utils import *
import os
from tqdm import tqdm
from glob import glob
import time
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    desc = "AnimeGANv3"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--model_dir', type=str, default='save_model/' + 'generated_Hayao',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--test_dir', type=str, default='dataset/test/test_photo',
                        help='Directory name of test photos')
    parser.add_argument('--save_dir', type=str, default='results/Hayao',
                        help='Path to save the generated files ')
    parser.add_argument('--if_adjust_brightness', type=bool, default=True,
                        help='adjust brightness by the real photo')

    """checking arguments"""

    return parser.parse_args()


def load_model(model_dir):
    print(f"DEBUG: Loading model: {model_dir}")
    model = tf.keras.models.load_model(model_dir)
    print(f'DEBUG: Model Loaded.')
    return model


def test(model_dir, save_dir, test_dir, if_adjust_brightness, img_size=[256, 256]):
    result_pred_dir = os.path.join(save_dir, 'predicted')
    result_comp_dir = os.path.join(save_dir, 'comparison')
    result_real_dir = os.path.join(save_dir, 'real_resized')
    check_create_folder(result_pred_dir)
    check_create_folder(result_comp_dir)
    check_create_folder(result_real_dir)
    test_files = glob('{}/*.*'.format(test_dir))

    test_generated = load_model(model_dir)

    begin = time.time()
    for sample_file in tqdm(test_files):

        loaded_img = load_test_data(sample_file, img_size)
        sample_image = np.asarray(loaded_img)

        fake_img = test_generated.predict(sample_image)

        real_save_path = os.path.join(result_real_dir, f'{os.path.basename(sample_file)}_real.jpg')
        save_images(sample_image, real_save_path, None)

        generated_save_path = os.path.join(result_pred_dir, f'{os.path.basename(sample_file)}_predicted.jpg')
        save_images(fake_img, generated_save_path, None)

        generated_comp_path = os.path.join(result_comp_dir, f'{os.path.basename(sample_file)}_real_pred.jpg')
        save_image_side_by_side_from_path(real_save_path, generated_save_path, generated_comp_path)

    end = time.time()
    print(f'test-time: {end - begin} s')
    print(f'one image test time : {(end - begin) / len(test_files)} s')





if __name__ == '__main__':
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        gpu0 = gpus[0]
        tf.config.experimental.set_memory_growth(gpu0, True)
        tf.config.set_visible_devices([gpu0], "GPU")

    arg = parse_args()
    print(arg.model_dir)
    test(arg.model_dir, arg.save_dir, arg.test_dir, arg.if_adjust_brightness)
