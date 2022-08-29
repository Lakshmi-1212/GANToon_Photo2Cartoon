from GANToon import GANToon
import argparse
from utils.utils import *


"""parsing and configuration"""


def parse_args():
    desc = "GANToon"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', type=str, default='Hayao', help='dataset_name')

    parser.add_argument('--epoch', type=int, default=101, help='The number of epochs to run')
    parser.add_argument('--init_epoch', type=int, default=10, help='The number of epochs for weight initialization')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='The size of batch size')  # if light : batch_size = 20
    parser.add_argument('--save_freq', type=int, default=1, help='The number of ckpt_save_freq')

    parser.add_argument('--init_lr', type=float, default=2e-4, help='Initialization phase generator learning rate')
    parser.add_argument('--g_lr', type=float, default=2e-5, help='Generator learning rate')
    parser.add_argument('--d_adv_lr', type=float, default=4e-5, help='Adversarial discriminator learning rate')
    parser.add_argument('--d_t_lr', type=float, default=4e-5, help='Textural discriminator learning rate')
    parser.add_argument('--ld', type=float, default=10.0, help='The gradient penalty lambda')

    parser.add_argument('--g_adv_weight', type=float, default=300.0, help='Weight about GAN')
    parser.add_argument('--d_adv_weight', type=float, default=300.0, help='Weight about GAN')
    parser.add_argument('--con_weight', type=float, default=1.5,
                        help='Weight about VGG19')

    parser.add_argument('--g_t_weight', type=float, default=10.0, help='Texture weight for generator')
    parser.add_argument('--d_t_weight', type=float, default=300.0, help='Texture weight for discriminator')
    parser.add_argument('--color_weight', type=float, default=10.,
                        help='Weight about color')
    parser.add_argument('--tv_weight', type=float, default=1.,
                        help='Weight about tv')

    parser.add_argument('--training_rate', type=int, default=1, help='training rate about G & D')
    parser.add_argument('--gan_type', type=str, default='lsgan',
                        help='[gan / lsgan / wgan-gp / wgan-lp / dragan / hinge')

    parser.add_argument('--img_size', type=list, default=[256, 256], help='The size of image: H and W')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_dis', type=int, default=3, help='The number of discriminator layer')
    parser.add_argument('--sn', type=str2bool, default=True, help='using spectral norm')

    parser.add_argument('--p_model_type', type=str, default='VGG19', help='Pretrained model to extract the feature map')


    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    return check_args(parser.parse_args())


"""checking arguments"""


def check_args(args):
    # --checkpoint_dir
    check_create_folder(args.checkpoint_dir)

    # --log_dir
    check_create_folder(args.log_dir)

    # --sample_dir
    check_create_folder(args.sample_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    gan = GANToon(args)
    gan.train()
    print(" [*] Training finished!")


if __name__ == '__main__':
    main()
