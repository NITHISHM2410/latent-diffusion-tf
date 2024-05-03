from samplers import DDPMSampler
from ldm.utils import *


class GenerateImages:
    def __init__(self, device, model, fs_model, latent_scale, ckpt_path, load_ckpt=True):
        """
        :param device: A tf.distribute.Strategy instance
        :param model: A trained diffusion model instance
        :param fs_model: A trained auto encoding model instance. Set to None if training DDPM instead of LDM.
        :param latent_scale: Latent space scaling value for LDM. Set to None if training DDPM instead of LDM.
        :param ckpt_path: Saved checkpoint path.
        :param load_ckpt: Whether to load ckpt upon class init.

        """
        self.device = device
        self.model = model
        self.fs_model = fs_model
        self.sampler = DDPMSampler(self.device)

        if self.fs_model:
            self.latent_scale = tf.Variable(latent_scale, trainable=False, name='scale')

        if load_ckpt:
            self.load_from_checkpoint(ckpt_path)

    def update_model(self, model):
        self.model = model

    def update_latent_scale(self, x):
        if self.fs_model:
            self.latent_scale.assign(x)
        else:
            raise Exception("No First stage found for LDM but called a function to update latent space scale.")

    def load_from_checkpoint(self, ckpt_path):
        """
        loads weights into the model from checkpoint path.
        :param ckpt_path: trained checkpoint path

        """
        if self.fs_model:
            ckpt = tf.train.Checkpoint(ema_model=self.model, latent_scale=self.latent_scale, fs_model=self.fs_model)
        else:
            ckpt = tf.train.Checkpoint(ema_model=self.model)
        ckpt.restore(ckpt_path).expect_partial()
        print("Checkpoint loaded.")

    @tf.function
    def fs_decode(self, x):
        if self.fs_model:
            if isinstance(x, tf.distribute.DistributedValues):
                return self.device.run(lambda p: self.fs_model.decode_ldm(1 / self.latent_scale * p, False), args=(x,))
            else:
                x = 1 / self.latent_scale * x
                return self.fs_model.decode_ldm(x, False)
        else:
            return x

    def apply_conditions(self, conditions, no_of):
        # Handle classes
        conditions = self.model.apply_conditioning(conditions)
        for cond in conditions:
            conditions[cond] = distribute_tensor(self.device, conditions[cond], no_of)
        return conditions

    def sample(self, no_of, conditions, hw=None):
        """

        Generates images.
        :param no_of: Number of images to generate. (64 - TPU / 16 - GPU)
        :param conditions: A dict of keys 'img_cond', 'text_cond', 'class_cond'. In case of Uncond modelling set that
        key to None or skip it.
        Input format for each condition is mentioned in models.py .

        :param hw: Output image resolution as tuple (H, W). if set to None, default size from model will be used.

        :return: Generated images.
        """
        # Sample Gaussian noise
        if hw:
            h, w = hw
        else:
            h, w = self.model.img_res, self.model.img_res

        images = tf.random.normal((no_of, h, w, self.model.c_in))
        images = distribute_tensor(self.device, images, no_of)

        conditions = self.apply_conditions(conditions, no_of)
        org_img = accumulate_dist(self.device, conditions['img_org'])
        del conditions['img_org']

        # Reverse diffusion for t time steps
        sampled = self.sampler.loop_on(self.model, images, conditions)
        sampled = self.fs_decode(sampled)
        sampled = accumulate_dist(self.device, sampled)

        # Set pixel values in display range
        sampled = tf.clip_by_value(sampled, 0, 1)

        # Accumulate distributed values
        img_cond = conditions['img_cond']
        img_cond = accumulate_dist(self.device, img_cond) if img_cond is not None else img_cond

        return sampled, img_cond, org_img
