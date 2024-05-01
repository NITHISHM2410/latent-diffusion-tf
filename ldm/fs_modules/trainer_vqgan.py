import os
import tensorflow as tf
from ldm.fs_modules.vqgan import LPIPS, Discriminator, VQGAN
from ldm.utils import save_grid, distribute_tensor, load_params_yaml, accumulate_dist, distribute_dataset
import gc


class Trainer:
    def __init__(self, config,
                 g_lr=5e-5, d_lr=5e-5, no_of=64, sampling_freq=5, is_logging=False, image_is_logging=True,
                 train_log_dir='logs/train_logs/', val_log_dir='logs/val_logs/', image_log_dir='logs/image_logs',
                 save_dir="save_dir", checkpoint_name='chk_point', lpips_path=None, device=None):
        """

        :param config: yaml file containing model params.
        :param g_lr: generator learning rate.
        :param d_lr: discriminator learning rate.
        :param no_of: number of images to reconstruct for every 'sampling_freq' epochs.
        :param sampling_freq: image sampling or reconstruction frequency in terms of epochs.
        :param is_logging: boolean, whether to log train/val results.
        :param image_is_logging: boolean, whether to log reconstructed images.
        :param train_log_dir: Directory to log train results.
        :param val_log_dir: Directory to log validation results.
        :param image_log_dir: Directory to log generated images.
        :param save_dir: Directory of the saved checkpoint. If no saved checkpoint available(no previous training) in
        the mentioned path then, one will be created during model training in the mentioned path.
        :param checkpoint_name: Name of the saved checkpoint. If no saved checkpoint available(no previous training) in
        above-mentioned path and name, one will be created during model training.
        :param lpips_path: path of pretrained LPIPS Model.
        :param device: A tf.distribute.Strategy instance.

        """
        self.lpips_path = lpips_path
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.device = device
        params = load_params_yaml(config)['params']

        with self.device.scope():
            self.disc_factor = params['trainer']['disc_factor']
            self.disc_start = params['trainer']['disc_start']
            self.codebook_weight = params['trainer']['codebook_weight']
            self.lpips_weight = params['trainer']['lpips_weight']

            self.model = VQGAN(**params['gen'])
            self.best_model = tf.keras.models.clone_model(self.model)
            self.best_model.build((None, self.model.img_size, self.model.img_size, self.model.c_in))

            self.lpips_loss = LPIPS(self.lpips_path)
            self.discriminator = Discriminator(**params['disc'])

            self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=self.g_lr)
            self.gen_optimizer.build(self.model.trainable_weights)

            self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=self.d_lr)
            self.disc_optimizer.build(self.discriminator.trainable_weights)

        self.train_stateful_metrics = ["ReconsLoss", "VQLoss", "GenLoss", "DWeight", "DiscRealOp", "DiscFakeOp"]
        self.val_stateful_metrics = ["ValReconsLoss", "ValVQLoss", "ValGenLoss", "ValDiscRealOp", "ValDiscFakeOp"]

        self.loss_trackers = dict()
        self.val_loss_trackers = dict()
        self.is_logging = is_logging
        self.image_is_logging = image_is_logging

        for lt in self.train_stateful_metrics:
            self.loss_trackers[lt] = tf.keras.metrics.Mean()
        for lt in self.val_stateful_metrics:
            self.val_loss_trackers[lt] = tf.keras.metrics.Mean()

        if self.is_logging:
            self.train_writer = tf.summary.create_file_writer(train_log_dir)
            self.val_writer = tf.summary.create_file_writer(val_log_dir)

        if self.image_is_logging:
            self.image_writer = tf.summary.create_file_writer(image_log_dir)

        self.train_counter = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.val_counter = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.img_log_counter = tf.Variable(0, trainable=False, dtype=tf.int32)

        self.checkpoint = None
        self.save_manager = None
        self.save_dir = save_dir
        self.checkpoint_name = checkpoint_name
        self.set_checkpoint(self.save_dir, self.checkpoint_name, None)

        self.no_of = no_of
        self.sampling_freq = sampling_freq
        self.best_loss = 9999999.0

    def set_checkpoint(self, save_dir, ckpt_name, cus_vars):
        """
        Points the checkpoint manager to a checkpoint in the specified directory.

        :param save_dir: Directory of the checkpoint to be restored or save.
        :param ckpt_name: Name of the checkpoint to be restored or save.
        :param cus_vars: Custom objects to be added to checkpoint instead of default objects.

        If no checkpoint is available in 'save_dir' named 'ckpt_name' then one will be created during training.
        If a checkpoint is available in the above-mentioned location, then it will be tracked. No new checkpoint will be created.

        """

        self.save_dir = save_dir
        self.checkpoint_name = ckpt_name

        if cus_vars is None:
            self.checkpoint = tf.train.Checkpoint(
                model=self.model,
                best_model=self.best_model,
                gen_opt=self.gen_optimizer,
                disc_opt=self.disc_optimizer,
                disc=self.discriminator.weights,
                train_counter=self.train_counter,
                val_counter=self.val_counter,
                img_log_counter=self.img_log_counter
            )
        else:
            self.checkpoint = tf.train.Checkpoint(**cus_vars)

        self.save_manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=save_dir,
            checkpoint_name=ckpt_name,
            max_to_keep=1
        )

    def restore_checkpoint(self, save_dir, ckpt_name, cus_vars):
        """
        Points the checkpoint manager to a checkpoint in the specified directory.

        :param save_dir: Directory of the checkpoint to be restored or save.
        :param ckpt_name: Name of the checkpoint to be restored or save.
        :param cus_vars: Custom objects to be added to checkpoint instead of default objects.

        If no checkpoint is available in 'save_dir' named 'ckpt_name' then one will be created during training.
        If a checkpoint is available in the above-mentioned location, then it will be tracked. No new checkpoint will be created.

        """
        self.set_checkpoint(save_dir, ckpt_name, cus_vars)
        ckpt_name = self.checkpoint_name + '-0'
        self.checkpoint.restore(os.path.join(self.save_dir, ckpt_name))
        if self.is_logging:
            print("Training logging done: ", self.train_counter.numpy())
            print("Validations logging done: ", self.val_counter.numpy())
        print("Optimizer iterations passed: ", self.gen_optimizer.iterations.numpy())

    @staticmethod
    def compute_scores(trackers):
        scores = []
        for lt in trackers:
            scores.append((lt, trackers[lt].result()))
        return scores

    @tf.function
    def distributed_model_call(self, kwargs):
        return self.device.run(self.model, kwargs=kwargs)

    @tf.function
    def distributed_decode_call(self, kwargs):
        return self.device.run(self.model.decode, kwargs=kwargs)

    @tf.function
    def train_step(self, data):
        def unit_step(x):
            x = x['image']
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                y, codebook_loss = self.model(x)

                recons_loss = tf.math.reduce_mean(
                    tf.math.abs(x - y) + self.lpips_loss(x, y) * self.lpips_weight
                )
                codebook_loss = tf.math.reduce_mean(codebook_loss) * self.codebook_weight

                fake_img_op = self.discriminator(y, training=False)
                gen_loss = -tf.math.reduce_mean(fake_img_op)

                disc_factor = tf.cond(
                    self.gen_optimizer.iterations >= self.disc_start,
                    true_fn=lambda: self.disc_factor,
                    false_fn=lambda: 0.0
                )
                d_weight = self.discriminator.calculate_adaptive_weights(
                    recons_loss=recons_loss,
                    gen_loss=gen_loss,
                    last_layer=self.model.get_last_layer()
                )

                total_gen_loss = (gen_loss * d_weight * disc_factor) + codebook_loss + recons_loss

                fake_img_op = self.discriminator(tf.stop_gradient(y), training=True)
                real_img_op = self.discriminator(tf.stop_gradient(x), training=True)

                disc_real = tf.math.reduce_mean(tf.nn.relu(1. - real_img_op))
                disc_fake = tf.math.reduce_mean(tf.nn.relu(1. + fake_img_op))
                total_disc_loss = 0.5 * (disc_real + disc_fake) * disc_factor

            gen_grads = gen_tape.gradient(total_gen_loss, self.model.trainable_weights)
            self.gen_optimizer.apply_gradients(zip(gen_grads, self.model.trainable_weights))

            disc_grads = disc_tape.gradient(total_disc_loss, self.discriminator.trainable_weights)
            self.disc_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_weights))

            return {
                'ReconsLoss': recons_loss,
                'VQLoss': codebook_loss,
                'GenLoss': gen_loss,
                'DWeight': d_weight,
                'DiscRealOp': tf.reduce_mean(real_img_op),
                'DiscFakeOp': tf.reduce_mean(fake_img_op),
            }

        results = self.device.run(unit_step, args=(next(data),))

        return {
            'ReconsLoss': self.device.reduce('MEAN', results['ReconsLoss'], axis=None),
            'VQLoss': self.device.reduce('MEAN', results['VQLoss'], axis=None),
            'GenLoss': self.device.reduce('MEAN', results['GenLoss'], axis=None),
            'DWeight': self.device.reduce('MEAN', results['DWeight'], axis=None),
            'DiscRealOp': self.device.reduce('MEAN', results['DiscRealOp'], axis=None),
            'DiscFakeOp': self.device.reduce('MEAN', results['DiscFakeOp'], axis=None),
        }

    @tf.function
    def test_step(self, data):
        def unit_step(x):
            x = x['image']
            y, codebook_loss = self.model(x)

            recons_loss = tf.math.reduce_mean(
                tf.math.abs(x - y) + self.lpips_loss(x, y) * self.lpips_weight
            )
            codebook_loss = tf.math.reduce_mean(codebook_loss) * self.codebook_weight

            fake_img_op = self.discriminator(y, training=False)
            gen_loss = -tf.math.reduce_mean(fake_img_op)

            fake_img_op = self.discriminator(y, training=False)
            real_img_op = self.discriminator(x, training=False)

            disc_real = tf.math.reduce_mean(tf.nn.relu(1. - real_img_op))
            disc_fake = tf.math.reduce_mean(tf.nn.relu(1. + fake_img_op))

            return {
                'ReconsLoss': recons_loss,
                'VQLoss': codebook_loss,
                'GenLoss': gen_loss,
                'DiscRealOp': tf.reduce_mean(real_img_op),
                'DiscFakeOp': tf.reduce_mean(fake_img_op),
            }

        results = self.device.run(unit_step, args=(next(data),))

        return {
            'ValReconsLoss': self.device.reduce('MEAN', results['ReconsLoss'], axis=None),
            'ValVQLoss': self.device.reduce('MEAN', results['VQLoss'], axis=None),
            'ValGenLoss': self.device.reduce('MEAN', results['GenLoss'], axis=None),
            'ValDiscRealOp': self.device.reduce('MEAN', results['DiscRealOp'], axis=None),
            'ValDiscFakeOp': self.device.reduce('MEAN', results['DiscFakeOp'], axis=None)
        }

    def sample(self, real_images, save_name, no_of, log_name):
        """

        :param real_images: set of real images to reconstruct from.
        :param save_name: path of reconstructed image to save.
        :param no_of: number of images to reconstruct.
        :param log_name: log name for images to save in tensorboard.
        """
        real_images = distribute_tensor(self.device, real_images[:no_of], no_of)
        recons_images, _ = self.distributed_model_call(
            kwargs={'inputs': real_images, 'training': tf.constant(False)})

        recons_images = accumulate_dist(self.device, recons_images)
        real_images = accumulate_dist(self.device, real_images)

        if self.image_is_logging:
            if log_name is None:
                name = "real(step 0) | recon(step 1) log.no {0}".format(str(self.img_log_counter.numpy()))
            else:
                name = "real(step 0) | recon(step 1) log.no {0}".format(str(log_name))

            with self.image_writer.as_default():
                tf.summary.image(name=name, data=real_images, max_outputs=no_of, step=0)
                tf.summary.image(name=name, data=recons_images, max_outputs=no_of, step=1)

            if log_name is None:
                self.img_log_counter.assign_add(self.sampling_freq)

        save_dir = "results/{0}_epoch".format(save_name)
        save_grid([real_images, recons_images], save_dir)
        return real_images, recons_images

    def train(self, epochs, train_ds, val_ds, train_steps, val_steps):
        """
        Training loop.

        :param epochs: number of epochs.
        :param train_ds: train tf.data.Dataset.
        :param val_ds: val tf.data.Dataset.
        :param train_steps: number of iterations per epoch for training data.
        :param val_steps: number of iterations per epoch for validation data.
        """

        train_ds = distribute_dataset(self.device, train_ds)
        sampling_ds = val_ds.unbatch().shuffle(self.no_of * 4).batch(self.no_of, True)

        for epoch in range(1, epochs + 1):
            train_data = iter(train_ds)
            val_data = val_ds
            
            print("Epoch {0}/{1}".format(epoch, epochs))

            for lt in self.train_stateful_metrics:
                self.loss_trackers[lt].reset_state()

            for lt in self.val_stateful_metrics:
                self.val_loss_trackers[lt].reset_state()

            progress = tf.keras.utils.Progbar(
                target=train_steps,
                stateful_metrics=self.train_stateful_metrics + self.val_stateful_metrics)

            for train_batch in range(train_steps):
                losses = self.train_step(train_data)

                for lt in self.train_stateful_metrics:
                    self.loss_trackers[lt].update_state(losses[lt])

                train_scores = self.compute_scores(self.loss_trackers)
                if self.is_logging:
                    with self.train_writer.as_default(step=self.train_counter.numpy()):
                        for score in train_scores:
                            tf.summary.scalar(score[0], score[1])
                    self.train_counter.assign_add(1)

                progress.update(train_batch + 1, values=train_scores, finalize=False)

            val_scores = self.evaluate(val_data=val_data, val_steps=val_steps, log_validation=True)
            progress.update(train_batch + 1, values=train_scores + val_scores, finalize=True)

            if epoch % self.sampling_freq == 0:
                for x in sampling_ds.take(1):
                    sample_img = x['image']
                self.sample(sample_img, epoch, self.no_of, None)

            if self.best_loss > self.val_loss_trackers['ValReconsLoss'].result():
                self.best_loss = self.val_loss_trackers['ValReconsLoss'].result()
                with self.device.scope():
                    for w1, w2 in zip(self.model.trainable_weights, self.best_model.trainable_weights):
                        w2.assign(w1)
                print("Best model captured....")

            self.save_manager.save(0)
            gc.collect()

    def evaluate(self, val_data, val_steps, log_validation=False):
        """
        Evaluation loop.

        :param val_data: Validation dataset.
        :param val_steps: number of iterations for validation data.
        :param log_validation:  boolean value, whether to log validation results.
        """
        val_data = iter(distribute_dataset(self.device, val_data))

        for lt in self.val_stateful_metrics:
            self.val_loss_trackers[lt].reset_state()

        log_validation = tf.constant(log_validation)
        for val_step in range(val_steps):
            losses = self.test_step(val_data)

            for lt in self.val_stateful_metrics:
                self.val_loss_trackers[lt].update_state(losses[lt])

            val_scores = self.compute_scores(self.val_loss_trackers)
            if log_validation and self.is_logging:
                with self.val_writer.as_default(step=self.val_counter.numpy()):
                    for score in val_scores:
                        tf.summary.scalar(score[0], score[1])
                self.val_counter.assign_add(1)

        return self.compute_scores(self.val_loss_trackers)
