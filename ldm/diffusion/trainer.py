import tensorflow as tf
from generate import GenerateImages
from ldm.utils import load_model_yaml, save_grid, distribute_dataset
import gc
import os


class Trainer:
    def __init__(self, model_config, lr=2e-5, ema_iterations_start=5000, loss_type='l2', no_of=64,
                 freq=3, sample_ema_only=True, is_logging=False, is_img_logging=True, device=None,
                 train_logdir="logs/train_logs/", val_logdir="logs/val_logs/", img_logdir="logs/img_logs/",
                 save_dir=None, checkpoint_name=None):

        """

        :param model_config: yaml file containing model params.
        :param lr: constant learning rate to train with.
        :param ema_iterations_start: no of iterations to start EMA.
        :param loss_type: 'l1' or 'l2' loss.
        :param no_of: no of images to generate at 'freq' frequency.
        :param freq: frequency of generating samples.
        :param sample_ema_only: Sample only from EMA model.
        :param device: A tf.distribute.Strategy instance.
        :param is_logging: Boolean value, whether to log results.
        :param is_img_logging: Boolean value, whether to log generated images.
        :param train_logdir: Directory to log train results.
        :param val_logdir: Directory to log validation results.
        :param img_logdir: Directory to log generated images.
        :param save_dir: Directory of the saved checkpoint. If no saved checkpoint available(no previous training) in
        the mentioned path then, one will be created during model training in the mentioned path.
        :param checkpoint_name: Name of the saved checkpoint. If no saved checkpoint available(no previous training) in
        above-mentioned path and name, one will be created during model training.

        """

        self.ema_iterations_start = ema_iterations_start
        self.no_of = no_of
        self.freq = freq
        self.sample_ema_only = sample_ema_only
        self.loss_type = loss_type
        self.lr = lr

        if not isinstance(device, tf.distribute.Strategy):
            raise Exception("Provide a tf.distribute.Strategy Instance")
        self.device = device

        with self.device.scope():
            # Initialize models
            self.model = load_model_yaml(model_config)
            self.ema_model = load_model_yaml(model_config)
            self.best_model = load_model_yaml(model_config)

            # Loss function MSE(l2) or MAE(l1)
            if loss_type == 'l2':
                self.base_loss = tf.keras.losses.MeanSquaredError(name="MSELoss",
                                                                  reduction=tf.keras.losses.Reduction.NONE)
            elif loss_type == 'l1':
                self.base_loss = tf.keras.losses.MeanAbsoluteError(name="MAELoss",
                                                                   reduction=tf.keras.losses.Reduction.NONE)
            else:
                raise Exception("provide l1 or l2 loss_fn")

            self.compute_loss = self.compute_loss

            # Optimizer
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
            self.optimizer.build(self.model.trainable_weights)
            self.ema_decay = tf.constant(0.99, dtype=tf.float32)

        # Train & Val step counter for tf.summary logging's.
        self.train_counter = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.val_counter = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.img_log_counter = tf.Variable(0, trainable=False, dtype=tf.int32)

        # Checkpoints
        self.checkpoint = None
        self.save_manager = None
        self.cur_trackable = None
        self.save_dir = save_dir
        self.checkpoint_name = checkpoint_name
        self.set_checkpoint(self.save_dir, self.checkpoint_name, None, True)

        # Loss trackers
        self.loss_tracker = tf.keras.metrics.Mean()
        self.val_loss_tracker = tf.keras.metrics.Mean()

        # Whether to perform logging
        self.is_logging = is_logging
        self.is_img_logging = is_img_logging
        if self.is_logging:
            self.train_logging = tf.summary.create_file_writer(train_logdir)
            self.val_logging = tf.summary.create_file_writer(val_logdir)

        if self.is_img_logging:
            self.img_logging = tf.summary.create_file_writer(img_logdir)

        # Image Generator
        self.generator = GenerateImages(
            self.device, self.ema_model, None, None,
            os.path.join(self.save_dir, self.checkpoint_name), False
        )

        # Initial best loss
        self.best_loss = 24.0

    def get_default_trackable(self, additional=None):
        if additional is None:
            additional = dict()
        variables = {
            'model': self.model,
            'ema_model': self.ema_model,
            'best_model': self.best_model,
            'optimizer': self.optimizer,
            'train_counter': self.train_counter,
            'val_counter': self.val_counter,
            'img_counter': self.img_log_counter,
        }
        variables.update(additional)
        return variables

    def set_checkpoint(self, save_dir, ckpt_name, cus_vars, replace):
        """
        Points the checkpoint manager to a checkpoint in the specified directory.

        :param save_dir: Directory of the checkpoint to be restored or save.
        :param ckpt_name: Name of the checkpoint to be restored or save.
        :param cus_vars: Custom objects to be added to checkpoint instead of default objects.
        :param replace: boolean, when True, custom vars replace default vars. when false, custom vars are appended to
         default vars.

        If no checkpoint is available in 'save_dir' named 'ckpt_name' then one will be created during training.
        If a checkpoint is available in the above-mentioned location, then it will be tracked. No new checkpoint will be created.

        """

        self.save_dir = save_dir
        self.checkpoint_name = ckpt_name

        if cus_vars is None:
            self.cur_trackable = self.get_default_trackable()
        else:
            if replace:
                self.cur_trackable = cus_vars
            else:
                self.cur_trackable = self.get_default_trackable(cus_vars)

        self.checkpoint = tf.train.Checkpoint(**self.cur_trackable)
        self.save_manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint,
            directory=save_dir,
            checkpoint_name=ckpt_name,
            max_to_keep=50
        )

    def restore_checkpoint(self, checkpoint_dir, checkpoint_name, custom_vars, replace):
        """
        Resumes training from checkpoint.

        :param checkpoint_dir: Directory of existing checkpoint.
        :param checkpoint_name: Name of the existing checkpoint in 'checkpoint_dir'.
        :param custom_vars: Custom objects to be added to checkpoint instead of default objects.
        :param replace: boolean, when True, custom vars replace default vars. when false, custom vars are appended to
         default vars.


        """
        self.set_checkpoint(checkpoint_dir, checkpoint_name, custom_vars, replace)
        ckpt_name = self.checkpoint_name + '-0'
        self.checkpoint.restore(os.path.join(self.save_dir, ckpt_name))
        if self.is_logging:
            print("Training logging done: ", self.train_counter.numpy())
            print("Validations logging done: ", self.val_counter.numpy())
        print("Optimizer iterations passed: ", self.optimizer.iterations.numpy())

    def compute_loss(self, y_true, y_pred):
        return tf.nn.compute_average_loss(tf.math.reduce_mean(self.base_loss(y_true, y_pred), axis=[1, 2]))

    def sample_time_step(self, size):
        return tf.experimental.numpy.random.randint(0, self.model.time_steps, size=(size,))[:, None]

    @tf.function
    def fs_encode(self, x):
        return x

    @tf.function
    def train_step(self, iterator):
        def unit_step(data):
            # Gather data
            image = self.fs_encode(data['image'])
            text_cond = data.get('text_cond', None)
            class_cond = data.get('class_cond', None)
            img_cond = data.get('img_cond', None)

            # Conditional guidance
            if self.model.class_cond:
                if tf.random.uniform(minval=0, maxval=1, shape=()) < 0.1:
                    class_cond = tf.fill(dims=tf.shape(class_cond), value=0)

            if self.model.text_cond:
                if tf.random.uniform(minval=0, maxval=1, shape=()) < 0.1:
                    text_cond = tf.fill(dims=tf.shape(text_cond), value=0.0)

            # Sample time step
            t = self.sample_time_step(size=tf.shape(image)[0])

            # Forward Noise
            noised_image, noise = self.model.forward_diff(image=image, time=t)

            # Forward pass
            inputs = {'batch': noised_image, 'time': t, 'class_cond': class_cond, 'text_cond': text_cond, 'img_cond': img_cond}
            inputs = self.model.apply_conditioning(inputs)
            del inputs['img_org'], inputs['cond_weight']

            with tf.GradientTape() as tape:
                output = self.model(inputs, training=True)
                loss = self.compute_loss(noise, output)

            # BackProp & Update
            grads = tape.gradient(loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

            # EMA
            if self.optimizer.iterations >= self.ema_iterations_start:
                for main_weights, ema_weights in zip(self.model.trainable_weights,
                                                     self.ema_model.trainable_weights):
                    ema_weights.assign(
                        ema_weights * self.ema_decay + main_weights * (1 - self.ema_decay)
                    )
            else:
                for main_weights, ema_weights in zip(self.model.trainable_weights,
                                                     self.ema_model.trainable_weights):
                    ema_weights.assign(main_weights)

            return loss

        # Distribute train batches across devices
        losses = self.device.run(unit_step, args=(next(iterator),))

        # Combine losses
        return self.device.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)

    # eval step using ema_model or main_model
    @tf.function
    def test_step(self, iterator, use_main):
        def unit_step(data):
            # Gather data
            image = self.fs_encode(data['image'])
            text_cond = data.get('text_cond', None)
            class_cond = data.get('class_cond', None)
            img_cond = data.get('img_cond', None)

            # Sample time step
            t = self.sample_time_step(size=tf.shape(image)[0])

            # Forward Noise
            noised_image, noise = self.model.forward_diff(image=image, time=t)

            # Forward pass
            inputs = {'batch': noised_image, 'time': t, 'class_cond': class_cond, 'text_cond': text_cond, 'img_cond': img_cond}
            inputs = self.model.apply_conditioning(inputs)
            del inputs['img_org'], inputs['cond_weight']

            if use_main:
                output = self.model(inputs, training=False)
            else:
                output = self.ema_model(inputs, training=False)

            # Compute loss
            loss = self.compute_loss(noise, output)
            return loss

        # Distribute train batches across devices
        losses = self.device.run(unit_step, args=(next(iterator),))

        # Combine losses
        return self.device.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)

    def sample(self, name, no_of, log_name, conditions, use_main=False):
        """
        Generates plots of generated images.

        :param name: image grid file name (Any string or int).
        :param no_of: number of images to generate.
        :param log_name: name for image logs in tensorboard. If set to None(generally during training),'img_log_counter'
         will be incremented and used for naming the image logs.
        :param conditions: Dict of conditions for conditional model. Include keys like 'class_cond', 'img_cond', 'text_cond'.
         View the models.py file to understand input format for each ldm model.

        :param use_main: boolean value, whether to use main model for generating.
        """
        # Update the sampler diffusion model and sample
        self.generator.update_model(self.model if use_main else self.ema_model)
        sampled, img_cond, org_img = self.generator.sample(no_of, conditions, None)

        # Log results in Tensorboard
        if self.is_img_logging:
            tb_name = "log.no {0}".format(log_name if log_name else str(self.img_log_counter.numpy()))

            with self.img_logging.as_default():
                t = 0
                for img in (sampled, img_cond, org_img):
                    if img is not None:
                        tf.summary.image(name=tb_name, data=img, max_outputs=no_of, step=t)
                        t += 1

            if log_name is None:
                self.img_log_counter.assign_add(self.freq)

        # Creating and saving sampled images plot
        img_save_dir = "results/{0}_epoch_{1}".format("main" if use_main else "ema", name)
        save_grid([org_img, img_cond, sampled], img_save_dir)

        return sampled

    def train(self, epochs, train_ds, val_ds, cond_ds, train_steps, val_steps):
        """
        Training loop.

        :param epochs: number of epochs.
        :param train_ds: train tf.data.Dataset.
        :param val_ds: val tf.data.Dataset.
        :param cond_ds: unbatched shuffled dataset containing conditional references as dict for sampling during
         training. If set to None, samples will be taken from val_ds.
        :param train_steps: number of iterations per epoch for training data.
        :param val_steps: number of iterations per epoch for validation data.
        """

        train_ds = distribute_dataset(self.device, train_ds)
        if not cond_ds and (not self.model.class_cond or not self.model.img_cond or not self.model.text_cond):
            cond_ds = val_ds
        cond_ds = cond_ds.unbatch().shuffle(self.no_of * 4).batch(self.no_of, True)

        for epoch in range(1, epochs + 1):
            # Make training and validation data iterable
            train_data = iter(train_ds)

            # Reset metrics states
            self.loss_tracker.reset_states()
            self.val_loss_tracker.reset_states()

            print("Epoch :", epoch)
            progbar = tf.keras.utils.Progbar(target=train_steps, stateful_metrics=['loss', 'val_loss'])

            for train_batch_no in range(train_steps):
                # train step
                train_loss = self.train_step(train_data)

                # update metrics
                self.loss_tracker.update_state(train_loss)
                progbar.update(train_batch_no + 1, values=[('loss', self.loss_tracker.result())], finalize=False)

                # log scores
                if self.is_logging:
                    with self.train_logging.as_default(self.train_counter.numpy()):
                        tf.summary.scalar("loss", self.loss_tracker.result())
                    self.train_counter.assign_add(1)

            # Validation process
            val_loss = self.evaluate(val_ds, val_steps, log_val_results=True, use_main=False)

            # Update scores
            progbar.update(train_batch_no + 1, values=[('loss', self.loss_tracker.result()), ('val_loss', val_loss)],
                           finalize=True)

            # Sampling images
            if epoch % self.freq == 0:
                for conditions in cond_ds.take(1):
                    del conditions['image']
                    self.sample(epoch, self.no_of, conditions=conditions, use_main=False, log_name=None)
                    if self.sample_ema_only is False:
                        self.sample(epoch, self.no_of, conditions=conditions, use_main=True, log_name=None)

            # Capturing best weights
            if self.val_loss_tracker.result() < self.best_loss:
                self.best_loss = self.val_loss_tracker.result()
                with self.device.scope():
                    for w1, w2 in zip(self.ema_model.trainable_weights, self.best_model.trainable_weights):
                        w2.assign(w1)
                print("Best model captured....")

            # Saving all weights
            self.save_manager.save(0)

            # Garbage Collect
            gc.collect()

    # eval function using main_model or ema_model
    def evaluate(self, val_data, val_steps, log_val_results=False, use_main=False):
        """
        Evaluation loop.

        :param val_data: Validation dataset.
        :param val_steps: number of iterations for validation data.
        :param log_val_results:  boolean value, whether to log validation results.
        :param use_main: boolean value, whether to use main model for generating.
        """
        val_data = iter(distribute_dataset(self.device, val_data))

        # Clear validation losses
        self.val_loss_tracker.reset_states()

        # Set val step params
        use_main = tf.constant(use_main)
        log_val_results = tf.constant(log_val_results)

        # Validation loop begins
        for _ in range(val_steps):
            val_loss = self.test_step(val_data, use_main)
            self.val_loss_tracker.update_state(val_loss)

            # logging based on trainer instance params
            if log_val_results and self.is_logging:
                with self.val_logging.as_default(self.val_counter.numpy()):
                    tf.summary.scalar("val_loss", self.val_loss_tracker.result())
                self.val_counter.assign_add(1)

        return self.val_loss_tracker.result()


class LDM_Trainer(Trainer):
    def __init__(self, device, fs_model_config, fs_latent_scale=1.0, fs_scale_by_std=True, fs_ckpt_path=None, *args,
                 **kwargs):
        """

        FIRST STAGE PARAMS
        :param device: A tf.distribute.Strategy instance.
        :param fs_model_config: yaml file containing auto encoding model params.
        :param fs_latent_scale: scaling value to be applied to latent space.
        :param fs_scale_by_std: Boolean, if True, scaling value be based on stddev on first batch,
         else 'fs_latent_scale' will be used.
        :param fs_ckpt_path: checkpoint path of AE model.

        """
        # Set Latent space scaling
        self.device = device
        self.scale_by_std = fs_scale_by_std
        self.latent_scale = tf.Variable(fs_latent_scale, trainable=False, name='scale')

        with self.device.scope():
            self.fs_model = load_model_yaml(fs_model_config)
        self.restore_fs_checkpoint(fs_ckpt_path)

        kwargs['device'] = self.device
        super().__init__(*args, **kwargs)

        self.generator = GenerateImages(
            self.device, self.ema_model, self.fs_model, self.latent_scale,
            os.path.join(self.save_dir, self.checkpoint_name), False
        )

    def get_default_trackable(self, additional=None):
        variables = super().get_default_trackable(additional)
        variables.update({'latent_scale': self.latent_scale, 'fs_model': self.fs_model})
        return variables

    def restore_fs_checkpoint(self, fs_ckpt_path):
        """
        Loads AE from checkpoint.

        :param fs_ckpt_path: Directory of existing AE checkpoint.0

        """
        fs_ckpt = tf.train.Checkpoint(model=self.fs_model)
        fs_ckpt.restore(fs_ckpt_path).expect_partial()
        self.fs_model.trainable = False
        print("FS(AE) Model loaded.")

    @tf.function
    def fs_encode(self, x):
        x = self.fs_model.encode_ldm(x, False)
        scale = tf.cond(
            self.scale_by_std and self.optimizer.iterations == 0,
            true_fn=lambda: 1 / tf.math.reduce_std(x),
            false_fn=lambda: self.latent_scale
        )
        self.latent_scale.assign(scale)
        return x * self.latent_scale

    def sample(self, name, no_of, log_name, conditions, use_main=False):
        self.generator.update_latent_scale(self.latent_scale)
        return super().sample(name, no_of, log_name, conditions, use_main)
