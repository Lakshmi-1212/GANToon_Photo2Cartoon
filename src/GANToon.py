from glob import glob
import gc
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from network.discriminator import Discriminator
from network.generator import Generator
from utils.data_loader import ImageGenerator
from utils.ops import *
from utils.utils import *


class GANToon(object):
    def __init__(self, args):
        config_defaults = {
            'real_loss_weight': 1.2,
            'fake_loss_weight': 1.2,
            'gray_loss_weight': 1.2,
            'real_blur_loss_weight': 0.8
        }
        self.model_name = 'AnimaTeGAN'
        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = args.log_dir
        self.dataset_name = args.dataset

        config_defaults['epoch'] = args.epoch
        config_defaults['init_epoch'] = args.init_epoch  # args.epoch // 20

        config_defaults['gan_type'] = args.gan_type
        config_defaults['batch_size'] = args.batch_size
        self.save_freq = args.save_freq

        config_defaults['init_lr'] = args.init_lr
        config_defaults['g_lr'] = args.g_lr
        config_defaults['d_adv_lr'] = args.d_adv_lr
        config_defaults['d_t_lr'] = args.d_t_lr

        """ Weight """
        config_defaults['g_adv_weight'] = args.g_adv_weight
        config_defaults['d_adv_weight'] = args.d_adv_weight

        config_defaults['g_t_weight'] = args.g_t_weight
        config_defaults['d_t_weight'] = args.d_t_weight
        config_defaults['con_weight'] = args.con_weight
        config_defaults['color_weight'] = args.color_weight
        config_defaults['tv_weight'] = args.tv_weight

        config_defaults['training_rate'] = args.training_rate
        config_defaults['ld'] = args.ld


        # Config is a variable that holds and saves hyperparameters and inputs
        self.config = config_defaults
        print(f'CONFIG: {self.config}')

        self.img_size = args.img_size
        self.img_ch = args.img_ch
        self.img_ch_texture = 1

        """ Discriminator """
        self.n_dis = args.n_dis
        self.ch = args.ch
        self.sn = args.sn

        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_create_folder(self.sample_dir)

        self.real_image_generator = ImageGenerator('./dataset/train_photo', self.img_size, self.config["batch_size"])
        self.cartoon_image_generator = ImageGenerator('./dataset/{}'.format(self.dataset_name + '/style'),
                                                      self.img_size,
                                                      self.config["batch_size"])
        self.cartoon_smooth_generator = ImageGenerator('./dataset/{}'.format(self.dataset_name + '/smooth'),
                                                       self.img_size, self.config["batch_size"])
        self.dataset_num = max(self.real_image_generator.num_images, self.cartoon_image_generator.num_images)

        self.p_model_type = args.p_model_type
        self.p_model = feature_map_model_init(self.p_model_type)

        print()
        print("##### Information #####")
        print("# gan type : ", self.config['gan_type'])
        print("# dataset : ", self.dataset_name)
        print("# max dataset number : ", self.dataset_num)
        print("# batch_size : ", self.config['batch_size'])
        print("# epoch : ", self.config['epoch'])
        print("# init_epoch : ", self.config['init_epoch'])
        print("# training image size [H, W] : ", self.img_size)
        print("# g_adv_weight,d_adv_weight,con_weight,g_t_weight,d_t_weight,color_weight,tv_weight : ", self.config['g_adv_weight'],
              self.config['d_adv_weight'], self.config['con_weight'], self.config['g_t_weight'], self.config['d_t_weight'], self.config['color_weight'],
              self.config['tv_weight'])
        print("# init_lr,g_lr,d_adv_lr,d_t_lr : ", self.config['init_lr'], self.config['g_lr'], self.config['d_adv_lr'], self.config['d_t_lr'])
        print(f"# training_rate G -- D: {self.config['training_rate']} : 1")
        print()

    ##################################################################################
    # Generator
    ##################################################################################

    def generator(self):
        G = Generator()
        G.build(input_shape=(None, self.img_size[0], self.img_size[1], self.img_ch))
        G.summary()
        return G

    ##################################################################################
    # Adversarial Discriminator
    ##################################################################################

    def discriminator_adv(self):
        D_adv = Discriminator(self.ch, self.n_dis, self.sn)
        D_adv.build(input_shape=(None, self.img_size[0], self.img_size[1], self.img_ch))
        D_adv.summary()
        return D_adv

    ##################################################################################
    # Textural Discriminator
    ##################################################################################

    def discriminator_t(self):
        D_t = Discriminator(self.ch, self.n_dis, self.sn)
        D_t.build(input_shape=(None, self.img_size[0], self.img_size[1], self.img_ch_texture))
        D_t.summary()
        return D_t

    ##################################################################################
    # Model
    ##################################################################################

    @tf.function
    def gradient_penalty(self, real, fake):
        if self.config['gan_type'].__contains__('dragan'):
            eps = tf.random.uniform(shape=tf.shape(real), minval=0., maxval=1.)
            _, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region

            fake = real + 0.5 * x_std * eps

        alpha = tf.random.uniform(shape=[self.config['batch_size'], 1, 1, 1], minval=0., maxval=1.)
        interpolated = real + alpha * (fake - real)

        disc_adv = self.discriminator_adv()
        disc_adv.build(input_shape=[None, self.img_size[0], self.img_size[1],
                                    self.img_ch])

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            logit, _ = disc_adv(interpolated)
        # gradient of D(interpolated)
        grad = tape.gradients(logit, interpolated)[0]
        grad_norm = tf.norm(tf.keras.Flatten(grad), axis=1)  # l2 norm

        GP = 0
        # WGAN - LP
        if self.config['gan_type'].__contains__('lp'):
            GP = self.config['ld'] * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.)))

        elif self.config['gan_type'].__contains__('gp') or self.config['gan_type'] == 'dragan':
            GP = self.config['ld'] * tf.reduce_mean(tf.square(grad_norm - 1.))

        return GP

    def train(self):

        """ Input Image"""
        # load_images returns iterator to a set of images based on the batch size
        real_img_op, cartoon_img_op, cartoon_smooth_op = self.real_image_generator.load_images(), \
                                                         self.cartoon_image_generator.load_images(), \
                                                         self.cartoon_smooth_generator.load_images()

        """ Define Generator, Discriminators """
        generated = self.generator()
        discriminator_adv = self.discriminator_adv()
        discriminator_t = self.discriminator_t()

        # summary writer
        self.writer = tf.summary.create_file_writer(self.log_dir + '/' + self.model_dir)

        """ Training """

        init_optim = Adam(self.config['init_lr'], beta_1=0.5, beta_2=0.999)
        G_optim = Adam(self.config['g_lr'], beta_1=0.5, beta_2=0.999)
        D_adv_optim = Adam(self.config['d_adv_lr'], beta_1=0.5, beta_2=0.999)
        D_t_optim = Adam(self.config['d_t_lr'], beta_1=0.5, beta_2=0.999)

        # saver to save model
        self.saver = tf.train.Checkpoint(generated=generated,
                                         discriminator_adv=discriminator_adv, discriminator_t=discriminator_t,
                                         G_optim=G_optim, D_adv_optim=D_adv_optim, D_t_optim=D_t_optim)

        # restore check-point if it exists
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = checkpoint_counter + 1
            print(f" [*] Load SUCCESS. Start epoch:{start_epoch + 1}")
        else:
            start_epoch = 0
            print(f" [!] Load failed... Start epoch:{start_epoch + 1}")

        init_mean_loss = []
        mean_loss = []
        j = self.config['training_rate']
        for epoch in range(start_epoch, self.config['epoch']):
            total_step = int(self.dataset_num / self.config['batch_size'])

            if epoch < self.config['init_epoch']:
                print(f'DEBUG: Start of epoch {epoch+1}/{self.config["init_epoch"]} - Initializing Generator')
            else:
                print(f'DEBUG: Start of epoch {epoch+1}/{self.config["epoch"]} - Training Network')

            with tqdm(range(total_step)) as tbar:
                for step in range(total_step):
                    real = next(real_img_op)[0]
                    cartoon = next(cartoon_img_op)[0]
                    cartoon_gray = next(cartoon_img_op)[1]
                    cartoon_smooth = next(cartoon_smooth_op)[0]

                    tbar.set_description(f'Epoch {epoch + 1}/{self.config["epoch"]}')
                    # Initialize generator with only content loss. Default init epoch = 10
                    if epoch < self.config['init_epoch']:
                        init_loss = self.init_train_step(generated, init_optim, epoch, real)
                        init_mean_loss.append(init_loss)
                        tbar.set_postfix(init_v_loss=init_loss.numpy(), mean_v_loss=np.mean(init_mean_loss))
                        tbar.update()
                        if (step + 1) % 200 == 0:
                            init_mean_loss.clear()
                    else:
                        d_adv_loss = 0
                        d_t_loss = 0
                        # Default training_rate = 1. No. of times Discriminators should be updated for every generator update.
                        if j == self.config['training_rate']:
                            # Update adversarial D network
                            d_adv_loss = self.d_adv_train_step(real, cartoon, cartoon_gray, cartoon_smooth,
                                                               generated, discriminator_adv, D_adv_optim, epoch)
                            # Update textural D network
                            d_t_loss = self.d_t_train_step(real, cartoon,
                                                           generated, discriminator_t, D_t_optim, epoch)

                        # Update G network
                        g_loss = self.g_train_step(real, cartoon_gray, generated,
                                                   discriminator_adv, discriminator_t, G_optim, epoch)


                        mean_loss.append([d_adv_loss, d_t_loss, g_loss])
                        if j == self.config['training_rate']:
                            tbar.set_postfix(d_adv_loss=d_adv_loss.numpy(),d_t_loss=d_t_loss.numpy(),
                                             g_loss=g_loss.numpy(),
                                             mean_d_adv_loss=np.mean(mean_loss, axis=0)[0],
                                             mean_d_t_loss=np.mean(mean_loss, axis=0)[1],
                                             mean_g_loss=np.mean(mean_loss, axis=0)[2])
                        else:
                            tbar.set_postfix(g_loss=g_loss.numpy(), mean_g_loss=np.mean(mean_loss, axis=0)[2])
                        tbar.update()

                        if (step + 1) % 200 == 0:
                            mean_loss.clear()

                        j = j - 1
                        if j < 1:
                            j = self.config['training_rate']

                gc.collect()

            if (epoch + 1) >= self.config['init_epoch'] and np.mod(epoch + 1, self.save_freq) == 0:
                    self.save(self.checkpoint_dir, epoch)

            if epoch >= self.config['init_epoch'] :

                # Save model
                save_model_path = 'save_model'
                if not os.path.exists(save_model_path):
                    os.makedirs(save_model_path)
                generated.save(os.path.join(save_model_path, 'generated_' + self.dataset_name), save_format='tf')
                print(f"DEBUG: Model saved: {os.path.join(save_model_path, 'generated_' + self.dataset_name)}")

                # Run model on validation data
                """ Result Image """
                val_files = glob('./dataset/{}/*.*'.format('val'))

                save_path = './{}/{:03d}/'.format(self.sample_dir, epoch+1)
                check_create_folder(save_path)

                val_images = []
                val_images_tensorboard = []
                for i, sample_file in enumerate(val_files):
                    sample_image = np.asarray(load_test_data(sample_file, self.img_size))
                    test_real = sample_image
                    test_generated_predict = generated.predict(test_real)
                    real_save_path = save_path + '{:03d}_real.jpg'.format(i)
                    generated_save_path = save_path + '{:03d}_gen.jpg'.format(i)
                    save_images(test_real, real_save_path , None)
                    save_images(test_generated_predict, generated_save_path, None)

                    save_image_side_by_side_from_path(real_save_path, generated_save_path, f'{save_path}{i:03d}.jpg')

                    val_images_tensorboard.append(test_generated_predict)

        gc.collect()

        print(f"DEBUG: End of train function")

    # Initialize generator with only content loss
    @tf.function
    def init_train_step(self, generated, init_optim, epoch, real):
        with tf.GradientTape() as tape:
            generator_images = generated(real)
            # init phase
            init_c_loss = compute_content_loss(self.p_model, self.p_model_type, real, generator_images)
            init_loss = self.config['con_weight'] * init_c_loss
        grads = tape.gradient(init_loss, generated.trainable_variables)
        init_optim.apply_gradients(zip(grads, generated.trainable_variables))

        with self.writer.as_default(step=epoch):
            """" Summary """
            tf.summary.scalar(name='G_init_loss', data=init_loss)

        return init_loss

    # Train the generator
    @tf.function
    def g_train_step(self, real, cartoon_gray, generated,
                     discriminator_adv, discriminator_t, G_optim, epoch):
        with tf.GradientTape() as tape:
            fake_image = generated(real)

            # Adversarial Loss - Ladv
            generated_adv_logit = discriminator_adv(fake_image)
            adv_loss = compute_generator_adv_loss(self.config['gan_type'], generated_adv_logit)

            # Content Loss - Lcon
            con_loss = compute_content_loss(self.p_model, self.p_model_type, real, fake_image)

            # Texture Loss - Ltexture
            textural_rep_list = textural_representation([fake_image], mode='uniform')
            generated_t_logit = discriminator_t(textural_rep_list[0])
            texture_loss = compute_generator_texture_loss(generated_t_logit)

            # Color Loss = Lcol
            col_loss = compute_color_loss(real, fake_image)

            # Total Variation Loss - Ltv
            tv_loss = compute_total_variation_loss(fake_image)

            # Total Loss
            total_gen_loss = self.config['g_adv_weight'] * adv_loss + \
                             self.config['con_weight'] * con_loss + \
                             self.config['g_t_weight'] * texture_loss + \
                             self.config['color_weight'] * col_loss + \
                             self.config['tv_weight'] * tv_loss

        grads = tape.gradient(total_gen_loss, generated.trainable_variables)
        G_optim.apply_gradients(zip(grads, generated.trainable_variables))

        with self.writer.as_default(step=epoch):
            """" Summary """
            tf.summary.scalar("Total Generator Loss", total_gen_loss)
            tf.summary.scalar("G_adv_loss", adv_loss)
            tf.summary.scalar("G_con_loss", con_loss)
            tf.summary.scalar("G_texture_loss", texture_loss)
            tf.summary.scalar("G_color_loss", col_loss)
            tf.summary.scalar("G_tv_loss", tv_loss)

        return total_gen_loss

    @tf.function
    def d_adv_train_step(self, real, cartoon, cartoon_gray, cartoon_smooth, generated, discriminator_adv,
                         D_adv_optim, epoch):

        with tf.GradientTape() as tape:
            fake_image = generated(real)
            d_cartoon_logit = discriminator_adv(cartoon)
            d_cartoon_gray_logit = discriminator_adv(cartoon_gray)
            d_cartoon_smooth_logit = discriminator_adv(cartoon_smooth)
            generated_logit = discriminator_adv(fake_image)
            """ Define Loss """
            if self.config['gan_type'].__contains__('gp') or self.config['gan_type'].__contains__('lp') or \
                    self.config['gan_type'].__contains__('dragan'):
                GP = self.gradient_penalty(real=cartoon, fake=fake_image)
            else:
                GP = 0.0

            d_adv_loss = compute_disc_adv_loss(self.config['gan_type'], d_cartoon_logit,
                                                                           d_cartoon_gray_logit,
                                                                           generated_logit,
                                                                           d_cartoon_smooth_logit, epoch, self.writer,
                                                                           self.config['real_loss_weight'],
                                                                           self.config['fake_loss_weight'],
                                                                           self.config['gray_loss_weight'],
                                                                           self.config['real_blur_loss_weight'])

            d_adv_total_loss = self.config['d_adv_weight'] * d_adv_loss + GP

        grads = tape.gradient(d_adv_total_loss, discriminator_adv.trainable_variables)
        D_adv_optim.apply_gradients(zip(grads, discriminator_adv.trainable_variables))

        with self.writer.as_default(step=epoch):
            """" Summary """
            tf.summary.scalar("d_adv_loss", d_adv_loss)
            tf.summary.scalar("Total Discriminator_adv_loss", d_adv_total_loss)

        return d_adv_total_loss


    @tf.function
    def d_t_train_step(self, real, cartoon, generated, discriminator_t,
                       D_t_optim, epoch):

        with tf.GradientTape() as tape:
            fake_image = generated(real)
            texture_rep_list = textural_representation([cartoon, fake_image], mode='uniform')

            real_cartoon_logit = discriminator_t(texture_rep_list[0])
            generated_logit = discriminator_t(texture_rep_list[1])

            """ Define Loss """
            disc_texture_loss = compute_disc_t_loss(real_cartoon_logit, generated_logit)
            total_d_t_loss = self.config['d_t_weight'] * disc_texture_loss

        grads = tape.gradient(total_d_t_loss, discriminator_t.trainable_variables)
        D_t_optim.apply_gradients(zip(grads, discriminator_t.trainable_variables))

        with self.writer.as_default(step=epoch):
            """" Summary """
            tf.summary.scalar("disc_texture_loss", disc_texture_loss)
            tf.summary.scalar("Total Discriminator_t_loss", total_d_t_loss)

        return total_d_t_loss

    @property
    def model_dir(self):
        return "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(self.model_name, self.dataset_name,
                                                   self.config["gan_type"],
                                                   int(self.config["g_adv_weight"]), int(self.config["d_adv_weight"]),
                                                   int(self.config["con_weight"]), int(self.config["d_t_weight"]),
                                                   int(self.config["color_weight"]), int(self.config["tv_weight"]))

    def save(self, checkpoint_dir, epoch):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        ckpt_manager = tf.train.CheckpointManager(self.saver, checkpoint_dir,
                                                  max_to_keep=5)
        ckpt_manager.save(checkpoint_number=epoch)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  # checkpoint file information

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(os.path.join(checkpoint_dir, ckpt_name)))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
