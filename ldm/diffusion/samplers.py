import tensorflow as tf
from tqdm import tqdm


class DDPMSampler:
    def __init__(self, device):
        self.device = device

    @tf.function
    def ddpm_diffuse(self, model, images, time, conditions):
        def diffuse_step(x, t, cond):
            bs = tf.shape(x)[0]
            t = tf.repeat(t, repeats=bs, axis=0)

            alpha = tf.gather(model.alphas, t)[:, None, None, None]
            beta = tf.gather(model.betas, t)[:, None, None, None]
            alpha_hat = tf.gather(model.alpha_hats, t)[:, None, None, None]

            t = t[:, None]

            inputs = cond.copy()

            cond_weight = inputs['cond_weight']
            del inputs['cond_weight']

            if not model.class_cond and not model.text_cond:
                inputs['batch'] = x
                inputs['time'] = t
                predicted_noise = model(inputs, training=False)

            else:
                inputs['batch'] = x
                inputs['time'] = t
                predicted_noise_cond = model(inputs, training=False)

                inputs['class_cond'] = tf.fill(tf.shape(inputs['class_cond']), value=0) if model.class_cond else None
                inputs['text_cond'] = tf.repeat(tf.constant([101, 102] + [0]*(model.text_seq_len-2))[None, :], bs, axis=0) if model.text_cond else None
                predicted_noise_uncond = model(inputs, training=False)

                predicted_noise = predicted_noise_uncond + cond_weight * (predicted_noise_cond - predicted_noise_uncond)

            if inputs['time'][0] > 0:
                noise = tf.random.normal(shape=tf.shape(x))
            else:
                noise = tf.zeros_like(x)

            x = (1 / tf.sqrt(alpha)) * (x - ((1 - alpha) / (tf.sqrt(1 - alpha_hat))) * predicted_noise) + tf.sqrt(beta) * noise
            return x

        output = self.device.run(diffuse_step, args=(images, time, conditions))
        return output

    def loop_on(self, model, images, conditions):
        for time in tqdm(reversed(tf.range(0, model.time_steps)), "Sampling Images...",
                         total=model.time_steps, leave=True, position=0):
            images = self.ddpm_diffuse(model, images, time, conditions)
        return images


class DDIMSampler(DDPMSampler):
    """
    Yet to be written.
    """

    def __init__(self, *args):
        super().__init__(*args)
